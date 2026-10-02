"""Served-scope candidate for a2aproject/A2A#2122 (section 8.4.1 rule 1).

These tests pin the candidate's behavior. Served scope is the reading favored
in that discussion, not a decided rule.
"""

import copy
import json

from typing import Any

import pytest

from a2a.types.a2a_pb2 import AgentCard
from a2a.utils import signing
from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.json_format import ParseDict


def _card() -> dict[str, Any]:
    return {
        'name': 'Probe',
        'description': 'A probe',
        'version': '1.0.0',
        'supportedInterfaces': [
            {
                'url': 'https://example.com/a2a',
                'protocolBinding': 'JSONRPC',
                'protocolVersion': '1.0',
            }
        ],
        'capabilities': {'streaming': True},
        'defaultInputModes': ['text/plain'],
        'defaultOutputModes': ['text/plain'],
        'skills': [{'id': 's', 'name': 'S', 'description': 'd', 'tags': ['t']}],
    }


def _canon(card: dict[str, Any]) -> dict[str, Any]:
    return json.loads(signing.canonicalize_served_agent_card(card))


def test_absent_required_field_stays_absent():
    card = _card()
    del card['description']
    assert 'description' not in _canon(card)


@pytest.mark.parametrize(
    ('path', 'value'),
    [(('description',), ''), (('skills',), []), (('defaultInputModes',), [])],
)
def test_present_required_field_at_default_is_kept(path, value):
    card = _card()
    card[path[0]] = value
    assert _canon(card)[path[0]] == value


def test_nested_required_field_at_default_is_kept():
    card = _card()
    card['skills'][0]['tags'] = []
    assert _canon(card)['skills'][0]['tags'] == []


def test_optional_keyword_field_present_at_default_is_kept():
    card = _card()
    card['documentationUrl'] = ''
    card['capabilities']['pushNotifications'] = False
    canon = _canon(card)
    assert canon['documentationUrl'] == ''
    assert canon['capabilities']['pushNotifications'] is False


def test_other_fields_at_default_are_dropped():
    card = _card()
    card['capabilities']['extensions'] = [
        {'uri': 'urn:x', 'required': False, 'description': ''}
    ]
    card['securityRequirements'] = [{}]
    card['skills'][0]['examples'] = []
    canon = _canon(card)
    assert canon['capabilities']['extensions'] == [{'uri': 'urn:x'}]
    assert 'securityRequirements' not in canon
    assert 'examples' not in canon['skills'][0]


def test_fields_outside_the_agent_card_schema_are_kept_as_served():
    card = _card()
    card['protocolVersion'] = '0.3.0'
    card['emptyList'] = []
    card['provider'] = {
        'url': 'https://example.com',
        'organization': 'O',
        'x': 1,
    }
    canon = _canon(card)
    assert canon['protocolVersion'] == '0.3.0'
    assert canon['emptyList'] == []
    assert canon['provider']['x'] == 1


def test_changing_an_unknown_field_changes_the_canonical_form():
    card = _card()
    card['url'] = 'https://a.example'
    before = _canon(card)
    card['url'] = 'https://b.example'
    assert _canon(card) != before


def test_signatures_are_excluded_and_input_is_not_mutated():
    card = _card()
    card['signatures'] = [{'protected': 'a', 'signature': 'b'}]
    before = copy.deepcopy(card)
    assert 'signatures' not in _canon(card)
    assert card == before


def test_matches_existing_canonical_form_when_no_field_is_at_default():
    card = _card()
    parsed = ParseDict(card, AgentCard())
    assert signing.canonicalize_served_agent_card(card) == (
        signing._canonicalize_agent_card(parsed)
    )


def _sign(card: dict[str, Any], key: Any, payload: str) -> dict[str, Any]:
    import jwt

    token = jwt.encode(
        json.loads(payload),
        key,
        algorithm='ES256',
        headers={'kid': 'k', 'typ': 'JOSE'},
    )
    protected, _, signature = token.split('.')
    signed = copy.deepcopy(card)
    signed['signatures'] = [{'protected': protected, 'signature': signature}]
    return signed


@pytest.fixture
def keypair():
    private = ec.generate_private_key(ec.SECP256R1())
    return private, private.public_key()


def _verifier(public_key: Any):
    return signing.create_served_card_signature_verifier(
        lambda kid, jku: public_key, ['ES256']
    )


def test_verifier_accepts_served_scope_signature(keypair):
    private, public = keypair
    card = _card()
    del card['description']
    card['skills'][0]['tags'] = []
    signed = _sign(card, private, signing.canonicalize_served_agent_card(card))
    _verifier(public)(signed)


def test_verifier_has_no_fallback_to_the_pruned_form(keypair):
    private, public = keypair
    card = _card()
    card['description'] = ''
    pruned = signing._canonicalize_agent_card(ParseDict(card, AgentCard()))
    signed = _sign(card, private, pruned)
    with pytest.raises(signing.InvalidSignaturesError):
        _verifier(public)(signed)


def test_verifier_rejects_a_parsed_card_that_differs_from_the_json(keypair):
    private, public = keypair
    card = _card()
    signed = _sign(card, private, signing.canonicalize_served_agent_card(card))
    other = ParseDict({**card, 'name': 'Other'}, AgentCard())
    with pytest.raises(signing.InvalidSignaturesError):
        _verifier(public)(signed, other)
    _verifier(public)(signed, ParseDict(signed, AgentCard()))


def test_verifier_rejects_a_card_with_no_signature(keypair):
    _, public = keypair
    with pytest.raises(signing.NoSignatureError):
        _verifier(public)(_card())


def test_verifier_rejects_a_tampered_card(keypair):
    private, public = keypair
    card = _card()
    signed = _sign(card, private, signing.canonicalize_served_agent_card(card))
    signed['name'] = 'Tampered'
    with pytest.raises(signing.InvalidSignaturesError):
        _verifier(public)(signed)
