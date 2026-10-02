import json

from collections.abc import Callable
from typing import Any, TypedDict

from google.api import field_behavior_pb2 as fb
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.json_format import MessageToDict, ParseDict, ParseError


try:
    import jwt

    from jwt import api_jws
    from jwt.api_jwk import PyJWK
    from jwt.exceptions import PyJWTError
    from jwt.utils import base64url_decode, base64url_encode
except ImportError as e:
    raise ImportError(
        'A2A Signing requires PyJWT to be installed. '
        'Install with: '
        "'pip install a2a-sdk[signing]'"
    ) from e

from a2a.types import AgentCard, AgentCardSignature
from a2a.utils._jcs import MAX_DEPTH, CanonicalizationError, canonicalize
from a2a.utils.proto_utils import _field_is_repeated


class SignatureVerificationError(Exception):
    """Base exception for signature verification errors."""


class NoSignatureError(SignatureVerificationError):
    """Exception raised when no signature is found on an AgentCard."""


class InvalidSignaturesError(SignatureVerificationError):
    """Exception raised when all signatures are invalid."""


class ProtectedHeader(TypedDict):
    """Protected header parameters for JWS (JSON Web Signature)."""

    kid: str
    """ Key identifier. """
    alg: str | None
    """ Algorithm used for signing. """
    jku: str | None
    """ JSON Web Key Set URL. """
    typ: str | None
    """ Token type.

    Best practice: SHOULD be "JOSE" for JWS tokens.
    """


def create_agent_card_signer(
    signing_key: PyJWK | str | bytes,
    protected_header: ProtectedHeader,
    header: dict[str, Any] | None = None,
) -> Callable[[AgentCard], AgentCard]:
    """Creates a function that signs an AgentCard and adds the signature.

    Args:
        signing_key: The private key for signing.
        protected_header: The protected header parameters.
        header: Unprotected header parameters.

    Returns:
        A callable that takes an AgentCard and returns the modified AgentCard with a signature.
    """

    def agent_card_signer(agent_card: AgentCard) -> AgentCard:
        """Signs agent card."""
        canonical_payload = _canonicalize_agent_card(agent_card)

        # The JWS payload has to be the canonical bytes themselves. Handing a
        # parsed dict to the JWT layer would let PyJWT re-serialize it with its
        # own json.dumps, which escapes non-ASCII, so the bytes signed would
        # differ from the bytes the verifier canonicalizes and checks.
        jws_string = api_jws.encode(
            payload=canonical_payload.encode('utf-8'),
            key=signing_key,
            algorithm=protected_header.get('alg', 'HS256'),
            headers=dict(protected_header),
        )

        # The result is a compact serialization: HEADER.PAYLOAD.SIGNATURE
        protected, _, signature = jws_string.split('.')

        agent_card_signature = AgentCardSignature(
            header=header,
            protected=protected,
            signature=signature,
        )

        agent_card.signatures.append(agent_card_signature)
        return agent_card

    return agent_card_signer


def create_signature_verifier(
    key_provider: Callable[[str | None, str | None], PyJWK | str | bytes],
    algorithms: list[str],
) -> Callable[[AgentCard], None]:
    """Creates a function that verifies the signatures on an AgentCard.

    The verifier succeeds if at least one signature is valid. Otherwise, it raises an error.

    Args:
        key_provider: A callable that accepts a key ID (kid) and a JWK Set URL (jku) and returns the verification key.
                      This function is responsible for fetching the correct key for a given signature.
        algorithms: A list of acceptable algorithms (e.g., ['ES256', 'RS256']) for verification used to prevent algorithm confusion attacks.

    Returns:
        A function that takes an AgentCard as input, and raises an error if none of the signatures are valid.
    """

    def signature_verifier(
        agent_card: AgentCard,
    ) -> None:
        """Verifies agent card signatures."""
        if not agent_card.signatures:
            raise NoSignatureError('AgentCard has no signatures to verify.')

        # The canonical form does not depend on which signature is being
        # checked, so it is computed once. A card with no canonical form has no
        # verifiable signature, and reporting that as InvalidSignaturesError
        # keeps every failure on this path a SignatureVerificationError.
        try:
            canonical_payload = _canonicalize_agent_card(agent_card)
        except CanonicalizationError as e:
            raise InvalidSignaturesError(
                'AgentCard cannot be canonicalized for verification'
            ) from e
        encoded_payload = base64url_encode(
            canonical_payload.encode('utf-8')
        ).decode('utf-8')

        for agent_card_signature in agent_card.signatures:
            try:
                # get verification key
                protected_header_json = base64url_decode(
                    agent_card_signature.protected.encode('utf-8')
                ).decode('utf-8')
                protected_header = json.loads(protected_header_json)
                kid = protected_header.get('kid')
                jku = protected_header.get('jku')
                verification_key = key_provider(kid, jku)

                token = f'{agent_card_signature.protected}.{encoded_payload}.{agent_card_signature.signature}'
                jwt.decode(
                    jwt=token,
                    key=verification_key,
                    algorithms=algorithms,
                )
                # Found a valid signature, exit the loop and function
                break
            except PyJWTError:
                continue
        else:
            # This block runs only if the loop completes without a break
            raise InvalidSignaturesError('No valid signature found')

    return signature_verifier


def _clean_empty(d: Any, depth: int = 0) -> Any:
    """Recursively remove empty strings, lists and dicts from a dictionary.

    Depth is bounded for the same reason canonicalization is: nesting reaches
    this function from `AgentExtension.params`, and without the bound a deeply
    nested card exhausts the interpreter stack here, before the canonicalizer
    ever gets the chance to reject it.
    """
    if depth > MAX_DEPTH:
        raise CanonicalizationError(
            f'nesting exceeds the maximum depth of {MAX_DEPTH}'
        )
    if isinstance(d, dict):
        cleaned_dict = {
            k: cleaned_v
            for k, v in d.items()
            if (cleaned_v := _clean_empty(v, depth + 1)) is not None
        }
        return cleaned_dict or None
    if isinstance(d, list):
        cleaned_list = [
            cleaned_v
            for v in d
            if (cleaned_v := _clean_empty(v, depth + 1)) is not None
        ]
        return cleaned_list or None
    if isinstance(d, str) and not d:
        return None
    return d


def _is_map(field: FieldDescriptor) -> bool:
    """Returns True if the field is a protobuf map."""
    message_type = field.message_type
    return message_type is not None and message_type.GetOptions().map_entry


def _is_well_known(descriptor: Descriptor | Any) -> bool:
    """Returns True for `google.protobuf` types, which carry free-form JSON."""
    return descriptor.full_name.startswith('google.protobuf.')


def _clean_field(value: Any, field: FieldDescriptor, depth: int) -> Any:
    """Removes empty values from the JSON form of one message field."""
    message_type = field.message_type
    if message_type is None or _is_well_known(message_type):
        return _clean_empty(value, depth)
    if _is_map(field):
        value_type = message_type.fields_by_name['value'].message_type
        if value_type is None or _is_well_known(value_type):
            return _clean_empty(value, depth)
        cleaned_map = {
            k: cleaned_v
            for k, v in value.items()
            if (cleaned_v := _clean_message(v, value_type, depth + 1))
        }
        return cleaned_map or None
    if _field_is_repeated(field):
        cleaned_list = [
            cleaned_v
            for v in value
            if (cleaned_v := _clean_message(v, message_type, depth + 1))
        ]
        return cleaned_list or None
    return _clean_message(value, message_type, depth) or None


def _clean_message(
    message_dict: dict[str, Any],
    descriptor: Descriptor | Any,
    depth: int = 0,
) -> dict[str, Any]:
    """Removes empty values from the JSON form of a message, by descriptor.

    `message_dict` is the `MessageToDict` output for a message of type
    `descriptor`. Walking the descriptor alongside the JSON keeps the field
    each value belongs to known at every level, which `_clean_empty` alone
    cannot tell. Free-form values (`google.protobuf.Struct` and friends) and
    keys the descriptor does not know fall back to `_clean_empty`.
    """
    if depth > MAX_DEPTH:
        raise CanonicalizationError(
            f'nesting exceeds the maximum depth of {MAX_DEPTH}'
        )
    fields = {field.json_name: field for field in descriptor.fields}
    cleaned: dict[str, Any] = {}
    for key, value in message_dict.items():
        field = fields.get(key)
        if field is None:
            cleaned_value = _clean_empty(value, depth + 1)
        else:
            cleaned_value = _clean_field(value, field, depth + 1)
        if cleaned_value is not None:
            cleaned[key] = cleaned_value
    return cleaned


def _canonicalize_agent_card(agent_card: AgentCard) -> str:
    """Canonicalizes the Agent Card JSON according to RFC 8785 (JCS)."""
    card_dict = MessageToDict(
        agent_card,
    )
    # Remove signatures field if present
    card_dict.pop('signatures', None)

    # Remove empty values, walking the AgentCard descriptor
    cleaned_dict = _clean_message(card_dict, AgentCard.DESCRIPTOR)
    return canonicalize(cleaned_dict or None)


# Candidate for a2aproject/A2A#2122, served-scope reading of section 8.4.1
# rule 1. It is the reading favored in that discussion, not a decided rule.
# Nothing above this line changes: the existing signer, verifier and
# canonicalization keep their behavior.
#
# Served scope applies rule 1 only to the fields present in the JSON being
# signed or verified. A REQUIRED field that is present stays even at its
# default value. A field declared with the `optional` keyword stays when it is
# present. Any other field at its default value is dropped. A field absent from
# the JSON is never added. Parsing into a protobuf message loses the
# difference between an absent field and one at its default, so this path
# works on the JSON as received.


def _is_required(field: FieldDescriptor) -> bool:
    """Returns True if the field carries google.api.field_behavior = REQUIRED."""
    return fb.REQUIRED in field.GetOptions().Extensions[fb.field_behavior]  # type: ignore[index]  # ty: ignore[invalid-argument-type]


def _has_optional_keyword(field: FieldDescriptor) -> bool:
    """Returns True for a scalar declared with the proto3 `optional` keyword."""
    return field.message_type is None and field.has_presence


def _is_scalar_default(value: Any, field: FieldDescriptor) -> bool:
    """Returns True if a JSON scalar equals the proto3 default for its field."""
    if field.type == FieldDescriptor.TYPE_ENUM:
        default = field.enum_type.values[0]
        return value in (default.name, default.number)
    if field.type == FieldDescriptor.TYPE_BOOL:
        return value is False
    if field.type in (FieldDescriptor.TYPE_STRING, FieldDescriptor.TYPE_BYTES):
        return value == ''
    return value in (0, '0') and not isinstance(value, bool)


def _served_required(value: Any, field: FieldDescriptor, depth: int) -> Any:
    """Keeps a REQUIRED field present in the served JSON, even at its default."""
    message_type = field.message_type
    if message_type is None or _is_well_known(message_type):
        if not _field_is_repeated(field):
            return value
        return [
            cleaned_v
            for v in value
            if (cleaned_v := _clean_empty(v, depth + 1)) is not None
        ]
    if _is_map(field):
        return _clean_field(value, field, depth) or {}
    if not _field_is_repeated(field):
        return _served_message(value, message_type, depth)
    return [
        cleaned_v
        for v in value
        if (cleaned_v := _served_message(v, message_type, depth + 1))
    ]


def _served_field(value: Any, field: FieldDescriptor, depth: int) -> Any:
    """Applies served-scope rule 1 to one field present in the served JSON."""
    message_type = field.message_type
    if value is None:
        result = None
    elif _is_required(field):
        result = _served_required(value, field, depth)
    elif _has_optional_keyword(field):
        result = value
    elif message_type is None and not _field_is_repeated(field):
        result = None if _is_scalar_default(value, field) else value
    elif _is_map(field):
        value_type = message_type.fields_by_name['value'].message_type
        if value_type is None or _is_well_known(value_type):
            result = _clean_field(value, field, depth)
        else:
            # Message values in a map go through the served-scope walker too,
            # so they get the same name handling as every other message.
            if not isinstance(value, dict):
                raise CanonicalizationError(
                    f'expected a JSON object for {field.full_name}'
                )
            result = {
                k: cleaned_v
                for k, v in value.items()
                if (cleaned_v := _served_message(v, value_type, depth + 1))
            } or None
    elif message_type is None or _is_well_known(message_type):
        result = _clean_field(value, field, depth)
    elif _field_is_repeated(field):
        result = [
            cleaned_v
            for v in value
            if (cleaned_v := _served_message(v, message_type, depth + 1))
        ] or None
    else:
        result = _served_message(value, message_type, depth) or None
    return result


def _served_message(
    message_dict: dict[str, Any],
    descriptor: Descriptor | Any,
    depth: int = 0,
) -> dict[str, Any]:
    """Applies served-scope rule 1 to a message as it appears in served JSON."""
    if depth > MAX_DEPTH:
        raise CanonicalizationError(
            f'nesting exceeds the maximum depth of {MAX_DEPTH}'
        )
    if not isinstance(message_dict, dict):
        raise CanonicalizationError(
            f'expected a JSON object for {descriptor.full_name}'
        )
    fields: dict[str, FieldDescriptor] = {}
    for field in descriptor.fields:
        fields[field.json_name] = field
        fields[field.name] = field
    cleaned: dict[str, Any] = {}
    seen_fields: dict[str, str] = {}
    for key, value in message_dict.items():
        field = fields.get(key)
        if field is not None:
            # A field given under both its JSON name and its proto name would
            # be kept twice in the canonical form while a typed parse keeps
            # only one, so two readers could see different values.
            earlier = seen_fields.get(field.full_name)
            if earlier is not None:
                raise CanonicalizationError(
                    f'field {field.full_name} appears as both '
                    f'{earlier!r} and {key!r}'
                )
            seen_fields[field.full_name] = key
        if field is None:
            # Not a field of this message. It is kept exactly as served, with
            # no default handling, so a passing signature covers it
            # (unknown-retain, discussed in a2aproject/A2A#2122).
            cleaned[key] = value
            continue
        cleaned_value = _served_field(value, field, depth + 1)
        if cleaned_value is not None:
            cleaned[key] = cleaned_value
    return cleaned


def canonicalize_served_agent_card(served_card: dict[str, Any]) -> str:
    """Canonicalizes an Agent Card as served, under the served-scope reading.

    `served_card` is the card JSON as received, before any protobuf parsing.
    `signatures` is excluded. Fields absent from the input stay absent.
    Fields the AgentCard schema does not define are kept exactly as served.
    """
    card = {k: v for k, v in served_card.items() if k != 'signatures'}
    return canonicalize(_served_message(card, AgentCard.DESCRIPTOR) or None)


def create_served_card_signature_verifier(
    key_provider: Callable[[str | None, str | None], PyJWK | str | bytes],
    algorithms: list[str],
) -> Callable[..., None]:
    """Creates a verifier for an Agent Card as served, under served scope.

    The returned function takes the card JSON exactly as received and,
    optionally, the AgentCard the caller parsed from it. Pass an unmodified
    copy of the JSON, since `parse_agent_card` changes its input in place.
    When both are given, the parsed card must equal a fresh parse of the JSON.

    This verifies the canonical representation selected by this candidate.
    Fields the AgentCard schema does not define are kept in it exactly as
    served, so a signature that passes covers them. The typed parse is used
    only to validate the card, never as canonicalization input. Only the
    served-scope form is tried. There is no fallback to another reading.
    """

    def served_card_verifier(
        served_card: dict[str, Any],
        agent_card: AgentCard | None = None,
    ) -> None:
        signatures = served_card.get('signatures') or []
        if not signatures:
            raise NoSignatureError('AgentCard has no signatures to verify.')
        try:
            parsed = ParseDict(
                served_card, AgentCard(), ignore_unknown_fields=True
            )
        except ParseError as e:
            raise InvalidSignaturesError(
                'served card does not parse as an AgentCard'
            ) from e
        if agent_card is not None and parsed != agent_card:
            raise InvalidSignaturesError(
                'parsed AgentCard does not match the served card'
            )
        try:
            canonical_payload = canonicalize_served_agent_card(served_card)
        except CanonicalizationError as e:
            raise InvalidSignaturesError(
                'AgentCard cannot be canonicalized for verification'
            ) from e
        encoded_payload = base64url_encode(
            canonical_payload.encode('utf-8')
        ).decode('utf-8')
        for signature in signatures:
            try:
                protected = signature['protected']
                header = json.loads(
                    base64url_decode(protected.encode('utf-8')).decode('utf-8')
                )
                verification_key = key_provider(
                    header.get('kid'), header.get('jku')
                )
                token = (
                    f'{protected}.{encoded_payload}.{signature["signature"]}'
                )
                jwt.decode(
                    jwt=token, key=verification_key, algorithms=algorithms
                )
                break
            except (PyJWTError, KeyError, TypeError, ValueError):
                continue
        else:
            raise InvalidSignaturesError('No valid signature found')

    return served_card_verifier
