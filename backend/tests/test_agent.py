import app.api.agent as agent_api
from app.agents.fraud_agent import (
    FraudAgent,
)
from app.main import app
from app.schemas.agent import (
    AgentAnalysisRequest,
)
from app.schemas.prediction import (
    FraudPrediction,
    TransactionAnalysisResponse,
)
from app.schemas.transaction import (
    TransactionAnalysisRequest,
)
from fastapi.testclient import TestClient


class FakeAnalyzeTransactionTool:
    """Deterministic production fraud-analysis tool for agent tests."""

    def run(
        self,
        transaction,
    ) -> TransactionAnalysisResponse:
        return TransactionAnalysisResponse(
            analysis_id="agent-test-analysis",
            created_at="2026-08-18T00:00:00+00:00",
            prediction=FraudPrediction(
                risk="HIGH",
                fraud_probability=0.91,
                threshold=0.83,
                model=(
                    "catboost_reduced_"
                    "fraud_detector"
                ),
                feature_count=20,
                decision_source=(
                    "catboost_ieee_cis"
                ),
                valid_output=True,
            ),
        )


class FakeMessage:
    content = (
        "The transaction should be reviewed "
        "because the production fraud detector "
        "classified it as high risk."
    )


class FakeChoice:
    message = FakeMessage()


class FakeCompletion:
    def __init__(self) -> None:
        self.choices = [
            FakeChoice()
        ]


class FakeCompletions:
    def create(
        self,
        **kwargs,
    ):
        return FakeCompletion()


class FakeChat:
    completions = FakeCompletions()


class FakeGroqClient:
    chat = FakeChat()


def transaction() -> TransactionAnalysisRequest:
    """Return one valid IEEE-CIS production transaction."""

    return TransactionAnalysisRequest(
        card2=404.0,
        card1=13926.0,
        addr1=315.0,
        C1=1.0,
        D2=10.0,
        C13=1.0,
        C2=1.0,
        M5_enc=1.0,
        D15=20.0,
        C5=0.0,
        C6=1.0,
        C14=1.0,
        M4_enc=0.0,
        purchaser_email_domain="gmail.com",
        card5=142.0,
        M6_enc=1.0,
        transaction_amt=250.0,
        log_amt=None,
        D10=12.0,
        D1=5.0,
    )


def transaction_payload() -> dict:
    """Return the API JSON representation of the test transaction."""

    return transaction().model_dump()


def test_agent_fallback_without_groq() -> None:
    agent = FraudAgent(
        analysis_tool=(
            FakeAnalyzeTransactionTool()
        ),
        groq_client=None,
    )

    result = agent.analyze(
        AgentAnalysisRequest(
            transaction=transaction()
        )
    )

    assert result.prediction.risk == "HIGH"

    assert (
        result.prediction.fraud_probability
        == 0.91
    )

    assert (
        result.prediction.decision_source
        == "catboost_ieee_cis"
    )

    assert result.llm_used is False
    assert result.agent_model is None

    assert (
        "CatBoost fraud detector"
        in result.explanation
    )

    assert (
        "91.00%"
        in result.explanation
    )

    assert len(
        result.recommendations
    ) == 3


def test_agent_with_groq_client() -> None:
    agent = FraudAgent(
        analysis_tool=(
            FakeAnalyzeTransactionTool()
        ),
        groq_client=FakeGroqClient(),
    )

    result = agent.analyze(
        AgentAnalysisRequest(
            transaction=transaction(),
            question=(
                "Why should this be reviewed?"
            ),
        )
    )

    assert result.prediction.risk == "HIGH"

    assert (
        result.prediction.threshold
        == 0.83
    )

    assert result.llm_used is True

    assert (
        "reviewed"
        in result.explanation.lower()
    )


def test_agent_status_endpoint() -> None:
    client = TestClient(
        app
    )

    response = client.get(
        "/api/agent"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"

    assert "groq_configured" in body
    assert "model" in body


def test_agent_analyze_endpoint() -> None:
    original_agent = (
        agent_api.fraud_agent
    )

    agent_api.fraud_agent = FraudAgent(
        analysis_tool=(
            FakeAnalyzeTransactionTool()
        ),
        groq_client=None,
    )

    try:
        client = TestClient(
            app
        )

        response = client.post(
            "/api/agent/analyze",
            json={
                "transaction": (
                    transaction_payload()
                )
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert (
            body["prediction"]["risk"]
            == "HIGH"
        )

        assert (
            body["prediction"][
                "fraud_probability"
            ]
            == 0.91
        )

        assert (
            body["prediction"][
                "decision_source"
            ]
            == "catboost_ieee_cis"
        )

        assert body["llm_used"] is False

    finally:
        agent_api.fraud_agent = (
            original_agent
        )


def test_agent_prompt_protects_anonymized_features() -> None:
    """Ensure the LLM receives explicit IEEE-CIS grounding rules."""

    captured = {}

    class CapturingCompletions:
        def create(
            self,
            **kwargs,
        ):
            captured.update(
                kwargs
            )

            return FakeCompletion()

    class CapturingChat:
        completions = (
            CapturingCompletions()
        )

    class CapturingGroqClient:
        chat = CapturingChat()

    agent = FraudAgent(
        analysis_tool=(
            FakeAnalyzeTransactionTool()
        ),
        groq_client=CapturingGroqClient(),
    )

    result = agent.analyze(
        AgentAnalysisRequest(
            transaction=transaction(),
            question=(
                "Is card1 the real card number "
                "and is addr1 the billing address?"
            ),
        )
    )

    assert result.llm_used is True

    messages = captured[
        "messages"
    ]

    system_content = (
        messages[0]["content"]
        .lower()
    )

    user_content = (
        messages[1]["content"]
        .lower()
    )

    assert "anonymized" in system_content
    assert "card1" in system_content
    assert "addr1" in system_content

    assert (
        "do not call card1/card2/card5 "
        "actual card numbers"
        in system_content
    )

    assert (
        "do not call addr1 a billing"
        in system_content
    )

    assert (
        "anonymized feature card1"
        in user_content
    )

    assert (
        "anonymized feature addr1"
        in user_content
    )

    assert (
        "do not describe them as actual "
        "card numbers"
        in user_content
    )
