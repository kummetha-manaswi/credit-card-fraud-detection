"""
Tests for FraudGuard AI customer assistant.
"""

import pytest
from src.assistant import get_assistant


def test_assistant_welcome_and_prompts():
    bot = get_assistant()
    welcome = bot.get_welcome_message()
    assert "FraudGuard AI" in welcome
    assert len(bot.QUICK_ACTIONS) >= 4


def test_assistant_security_alert():
    bot = get_assistant()
    resp = bot.respond("Can I give you my card PIN and CVV?")
    assert "Never share your card PIN, CVV" in resp


def test_assistant_contextual_high_risk():
    bot = get_assistant()
    sample_result = {
        "risk_level": "HIGH",
        "fraud_probability": 0.985,
        "risk_score": 98.5,
        "action": "BLOCK & INVESTIGATE"
    }
    resp = bot.respond("What does this result mean?", current_page="🔍 Risk Checker", last_eval_result=sample_result)
    assert "High Risk Assessment" in resp
    assert "BLOCK & INVESTIGATE" in resp
    assert "98.50%" in resp


def test_assistant_contextual_low_risk():
    bot = get_assistant()
    sample_result = {
        "risk_level": "LOW",
        "fraud_probability": 0.005,
        "risk_score": 0.5,
        "action": "APPROVE"
    }
    resp = bot.respond("Why was my transaction result low risk?", current_page="🔍 Risk Checker", last_eval_result=sample_result)
    assert "Low Risk Approved" in resp
    assert "APPROVE" in resp


def test_assistant_navigation_questions():
    bot = get_assistant()
    r1 = bot.respond("Where can I check a transaction?")
    assert "Risk Checker" in r1

    r2 = bot.respond("Where can I see fraud patterns?")
    assert "Fraud Intelligence" in r2

    r3 = bot.respond("Why did AI make this decision?")
    assert "Explainable AI" in r3


def test_assistant_customer_questions():
    bot = get_assistant()
    r1 = bot.respond("I don't recognize a transaction on my card")
    assert "Unrecognized Transaction" in r1

    r2 = bot.respond("My payment was declined")
    assert "Declined Payment" in r2

    r3 = bot.respond("What is a fraud alert?")
    assert "fraud alert" in r3.lower()
