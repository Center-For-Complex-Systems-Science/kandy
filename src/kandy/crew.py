from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class Kandy():
    """KANDy research automation crew.

    Orchestrates experimental result cleaning, new experiment generation,
    Python package development, and journal-quality review for the
    Kolmogorov-Arnold Networks for Dynamics (KANDy) project.

    Process: Hierarchical — the PhD Advisor manages and delegates to
    PhD Student, Professional Coder, and Journal Reviewer agents.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # ------------------------------------------------------------------ #
    # Manager agent (not decorated — set explicitly as manager_agent)     #
    # ------------------------------------------------------------------ #

    def phd_advisor(self) -> Agent:
        """Principal Investigator who orchestrates the research program."""
        return Agent(
            config=self.agents_config['phd_advisor'],  # type: ignore[index]
            verbose=True,
        )

    # ------------------------------------------------------------------ #
    # Worker agents                                                        #
    # ------------------------------------------------------------------ #

    @agent
    def phd_student(self) -> Agent:
        """Runs experiments, cleans results, and extracts symbolic formulas."""
        return Agent(
            config=self.agents_config['phd_student'],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def professional_coder(self) -> Agent:
        """Transforms research notebooks into a clean Python package."""
        return Agent(
            config=self.agents_config['professional_coder'],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def journal_reviewer(self) -> Agent:
        """Provides peer-review-quality feedback on results and claims."""
        return Agent(
            config=self.agents_config['journal_reviewer'],  # type: ignore[index]
            verbose=True,
        )

    # ------------------------------------------------------------------ #
    # Tasks                                                                #
    # ------------------------------------------------------------------ #

    @task
    def clean_experimental_results(self) -> Task:
        return Task(
            config=self.tasks_config['clean_experimental_results'],  # type: ignore[index]
            output_file='outputs/audit_report.md',
        )

    @task
    def generate_new_results(self) -> Task:
        return Task(
            config=self.tasks_config['generate_new_results'],  # type: ignore[index]
            output_file='outputs/experiment_{system}.py',
        )

    @task
    def develop_package_code(self) -> Task:
        return Task(
            config=self.tasks_config['develop_package_code'],  # type: ignore[index]
            output_file='outputs/package_design.md',
        )

    @task
    def review_results(self) -> Task:
        return Task(
            config=self.tasks_config['review_results'],  # type: ignore[index]
            output_file='outputs/review.md',
        )

    # ------------------------------------------------------------------ #
    # Crew                                                                 #
    # ------------------------------------------------------------------ #

    @crew
    def crew(self) -> Crew:
        """Creates the KANDy crew with hierarchical process.

        The PhD Advisor acts as manager and dynamically delegates tasks to
        the PhD Student, Professional Coder, and Journal Reviewer.
        """
        return Crew(
            agents=self.agents,          # [phd_student, professional_coder, journal_reviewer]
            tasks=self.tasks,            # Populated automatically by @task decorators
            process=Process.hierarchical,
            manager_agent=self.phd_advisor(),
            verbose=True,
        )
