from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from .tools.kandy_tools import (
    KANDyAPITool,
    ReadFileTool,
    RunScriptTool,
    WriteScriptTool,
)

# Shared tool instances (stateless — safe to share across agents)
_api_tool   = KANDyAPITool()
_read_tool  = ReadFileTool()
_run_tool   = RunScriptTool()
_write_tool = WriteScriptTool()

# ---------------------------------------------------------------------------
# Runtime limits
#   max_iter          — caps the internal tool-use loop per task
#   max_execution_time — hard wall-clock cap in seconds per task
#   respect_context_window — auto-summarize when context window fills up
# ---------------------------------------------------------------------------
_RESEARCH_LIMITS = dict(
    max_iter=10,
    max_execution_time=1800,   # 30 min per research task
    respect_context_window=True,
)
_REVIEW_LIMITS = dict(
    max_iter=8,
    max_execution_time=900,    # 15 min per review / feedback task
    respect_context_window=True,
)
_CODER_LIMITS = dict(
    max_iter=8,
    max_execution_time=600,    # 10 min
    respect_context_window=True,
)


@CrewBase
class Kandy():
    """KANDy research automation crew.

    Task order (context-DAG):
      Phase 1 (parallel): research_kane_lift, fix_kuramoto_model,
                          fix_ikeda_model, write_baselines
      Phase 2:            advisor_feedback   (reads Phase 1 outputs)
      Phase 3:            write_run_report   (what worked / didn't)
      Phase 4:            review_results     (formal peer review)

    Process: Hierarchical — phd_advisor manages and provides real-time
    feedback during delegation.  advisor_feedback task captures written
    per-agent notes that persist via long-term memory into the next run.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ #
    # Manager agent (not decorated — passed as manager_agent)            #
    # ------------------------------------------------------------------ #

    def phd_advisor(self) -> Agent:
        """Principal Investigator — orchestrates and provides real-time feedback."""
        return Agent(
            config=self.agents_config['phd_advisor'],   # type: ignore[index]
            memory=True,
            verbose=True,
            **_REVIEW_LIMITS,
        )

    # ------------------------------------------------------------------ #
    # Worker agents                                                        #
    # ------------------------------------------------------------------ #

    @agent
    def kane_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['kane_researcher'],   # type: ignore[index]
            tools=[_api_tool, _read_tool, _write_tool, _run_tool],
            memory=True,
            verbose=True,
            **_RESEARCH_LIMITS,
        )

    @agent
    def kandy_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['kandy_researcher'],  # type: ignore[index]
            tools=[_api_tool, _read_tool, _write_tool, _run_tool],
            memory=True,
            verbose=True,
            **_RESEARCH_LIMITS,
        )

    @agent
    def baseline_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['baseline_researcher'],   # type: ignore[index]
            tools=[_api_tool, _read_tool, _write_tool, _run_tool],
            memory=True,
            verbose=True,
            **_RESEARCH_LIMITS,
        )

    @agent
    def professional_coder(self) -> Agent:
        return Agent(
            config=self.agents_config['professional_coder'],    # type: ignore[index]
            tools=[_read_tool, _write_tool],
            memory=True,
            verbose=True,
            **_CODER_LIMITS,
        )

    @agent
    def journal_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['journal_reviewer'],  # type: ignore[index]
            tools=[_read_tool],
            memory=True,
            verbose=True,
            **_REVIEW_LIMITS,
        )

    # ------------------------------------------------------------------ #
    # Tasks — Phase 1: parallel research                                  #
    # ------------------------------------------------------------------ #

    @task
    def research_kane_lift(self) -> Task:
        return Task(
            config=self.tasks_config['research_kane_lift'],     # type: ignore[index]
            output_file='outputs/kane_report.md',
        )

    @task
    def fix_kuramoto_model(self) -> Task:
        return Task(
            config=self.tasks_config['fix_kuramoto_model'],     # type: ignore[index]
            output_file='outputs/kuramoto_fixed.py',
        )

    @task
    def fix_ikeda_model(self) -> Task:
        return Task(
            config=self.tasks_config['fix_ikeda_model'],        # type: ignore[index]
            output_file='outputs/ikeda_fixed.py',
        )

    @task
    def write_baselines(self) -> Task:
        return Task(
            config=self.tasks_config['write_baselines'],        # type: ignore[index]
            output_file='outputs/baselines/comparison_table.md',
        )

    # ------------------------------------------------------------------ #
    # Tasks — Phase 2: feedback                                           #
    # ------------------------------------------------------------------ #

    @task
    def advisor_feedback(self) -> Task:
        return Task(
            config=self.tasks_config['advisor_feedback'],       # type: ignore[index]
            output_file='outputs/advisor_feedback.md',
        )

    # ------------------------------------------------------------------ #
    # Tasks — Phase 3/4: reporting                                        #
    # ------------------------------------------------------------------ #

    @task
    def write_run_report(self) -> Task:
        return Task(
            config=self.tasks_config['write_run_report'],       # type: ignore[index]
            output_file='outputs/run_report.md',
        )

    @task
    def review_results(self) -> Task:
        return Task(
            config=self.tasks_config['review_results'],         # type: ignore[index]
            output_file='outputs/review.md',
        )

    # ------------------------------------------------------------------ #
    # Crew                                                                 #
    # ------------------------------------------------------------------ #

    @crew
    def crew(self) -> Crew:
        """KANDy research crew — hierarchical, PhD Advisor manages."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.phd_advisor(),
            memory=True,
            max_rpm=20,       # throttle API calls across the whole crew
            verbose=True,
        )
