from types import SimpleNamespace

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
    def run(
        self,
        transaction,
    ) -> TransactionAnalysisResponse:
        return TransactionAnalysisResponse(
            analysis_id="analysis-123",
            created_at=(
                "2026-08-17T00:00:00+00:00"
            ),
            prediction=FraudPrediction(
                risk="HIGH",
                model=(
                    "Qwen/Qwen2.5-1.5B-Instruct"
                ),
                adapter=(
                    "ebbejulankapalli/"
                    "financial-fraud-detector-"
                    "qwen2.5-qlora"
                ),
                decision_source=(
                    "fine_tuned_llm"
                ),
                raw_output="HIGH",
                valid_output=True,
            ),
        )


def transaction() -> TransactionAnalysisRequest:
    return TransactionAnalysisRequest(
        type="TRANSFER",
        amount=85000,
        oldbalanceOrg=85000,
        newbalanceOrig=0,
        oldbalanceDest=0,
        newbalanceDest=85000,
    )


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
    assert result.llm_used is False
    assert result.agent_model is None
    assert len(result.recommendations) == 3


class FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["temperature"] == 0.2

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "The classifier marked "
                            "this transaction HIGH risk."
                        )
                    )
                )
            ]
        )


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


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

    assert result.llm_used is True
    assert result.prediction.risk == "HIGH"

    assert "HIGH risk" in (
        result.explanation
    )


def test_agent_status_endpoint() -> None:
    client = TestClient(app)

    response = client.get(
        "/api/agent"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert "groq_configured" in body


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
        client = TestClient(app)

        response = client.post(
            "/api/agent/analyze",
            json={
                "transaction": {
                    "type": "TRANSFER",
                    "amount": 85000,
                    "oldbalanceOrg": 85000,
                    "newbalanceOrig": 0,
                    "oldbalanceDest": 0,
                    "newbalanceDest": 85000,
                }
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert (
            body["prediction"]["risk"]
            == "HIGH"
        )

        assert body["llm_used"] is False

    finally:
        agent_api.fraud_agent = (
            original_agent
        )
