from typing import Literal

from pydantic import BaseModel, Field, field_validator

AgentName = Literal["sales", "knowledge", "economic"]

AgentStatus = Literal[
    "completed",
    "needs_input",
    "failed",
]


class AgentTask(BaseModel):
    """Contrato de tarefa enviado pelo Supervisor ao especialista."""

    request_id: str
    agent: AgentName
    objective: str
    question: str


class AgentResult(BaseModel):
    """Contrato padronizado de retorno dos agentes."""

    request_id: str
    agent: AgentName
    status: AgentStatus
    summary: str
    evidence: str = ""
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    """Decisão estruturada do Supervisor."""

    agents: list[AgentName] = Field(
        min_length=1,
        max_length=3,
        description=("Agentes que devem participar: sales, knowledge e/ou economic."),
    )
    rationale: str = Field(description="Justificativa objetiva da seleção dos agentes.")

    @field_validator("agents")
    @classmethod
    def agents_must_be_unique(
        cls,
        agents: list[AgentName],
    ) -> list[AgentName]:
        if len(agents) != len(set(agents)):
            raise ValueError("A lista de agentes não pode conter duplicatas.")
        return agents
