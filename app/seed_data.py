from __future__ import annotations

from app.schemas import ProjectRecord, TemplateRecord, TemplateSummary, TemplateType, UserProfile


DEFAULT_TEMPLATES: dict[str, TemplateRecord] = {
    "geeksvisor_classic": TemplateRecord(
        template_id="geeksvisor_classic",
        label="GeeksVisor Classic",
        description="Balanced and portfolio-led proposal for general SaaS and full-stack jobs.",
        best_for="General full-stack, SaaS, and product-engineering proposals.",
        body=(
            "Hi, this sounds like a perfect fit.\n\n"
            "I worked on [ProjectName], [brief context], where I [specific relevant achievement]. "
            "Alongside that, I’ve delivered 30+ SaaS and AI apps as part of the GeeksVisor team.\n\n"
            "Some examples relevant to your vision:\n"
            "[Project1] -> [Brief description] ([Tech stack])\n"
            "[Project2] -> [Brief description] ([Tech stack])\n\n"
            "Portfolio Glimpse:\n"
            "saddlefit.io | ledgeriq.ai | wealthbuilder.io | earlybirdee.co | aplusresumes.ai\n\n"
            "I’m comfortable owning features end-to-end, from clean UI to reliable backend APIs. "
            "Would you prefer to share details on chat, or schedule a quick call to align?\n\n"
            "Best,\n[Name]"
        ),
    ),
    "consultative_expert": TemplateRecord(
        template_id="consultative_expert",
        label="Consultative Expert",
        description="Diagnosis-first technical style for AI, backend, and architecture-heavy jobs.",
        best_for="AI, RAG, backend, AWS, and complex systems proposals.",
        body=(
            "Hi, this aligns closely with my recent project.\n\n"
            "I read your description about [Specific Problem], and it reminds me of a hurdle I solved in [ProjectName]. "
            "Most people skip [Specific Technical Detail], but I usually handle it by [Technical Solution].\n\n"
            "Let me share the technical details of [ProjectName]:\n"
            "Project Overview: [2-3 sentences from context]\n"
            "Required Tasks:\n"
            "- [Task 1]\n"
            "- [Task 2]\n"
            "Technologies: [Tech Stack]\n\n"
            "I’m comfortable working inside existing AWS setups and managing complex data flows. "
            "Do you already have a preferred tech stack for this, or are you looking for a recommendation?\n\n"
            "Best,\n[Name]\n[AWS / technical credential if relevant]"
        ),
    ),
    "the_fast_mover": TemplateRecord(
        template_id="the_fast_mover",
        label="The Fast Mover",
        description="Short, direct, and urgency-friendly proposal style.",
        best_for="Fixed-price, urgent delivery, or clients who want immediate momentum.",
        body=(
            "I can definitely help here.\n\n"
            "I’ve built systems exactly like this (see [ProjectName]) where I [Result]. "
            "I can jump into your repository today and start delivering immediately.\n\n"
            "Project Details:\n"
            "- [Achievement 1]\n"
            "- [Achievement 2]\n"
            "- [Achievement 3]\n"
            "Technologies: [Inline list]\n\n"
            "Quick Portfolio: saddlefit.io | ledgeriq.ai | earlybirdee.co\n\n"
            "I build systems that are easy for others to work with - clear patterns and simple extension points. "
            "Are you available for a 5-minute chat to finalize the requirements?\n\n"
            "Best,\n[Name]"
        ),
    ),
}


DUMMY_PROJECTS: list[ProjectRecord] = [
    ProjectRecord(
        project_id="p1",
        title="StoryBloom - AI Storybook Generator",
        description=(
            "A serverless application that uses Node.js and Gemini APIs to generate children's stories. "
            "It ensures character consistency across generated images and integrates with the LuLu API for physical printing."
        ),
        tech_stack=["Node.js", "Gemini Nano Banana", "OpenAI", "LuLu API", "Serverless"],
        role="Lead Generative AI Developer",
    ),
    ProjectRecord(
        project_id="p2",
        title="Aha-doc - Document Intelligence",
        description=(
            "An AI-powered document analysis platform that uses RAG (Retrieval-Augmented Generation) "
            "to help users query complex PDF documents and extract structured insights."
        ),
        tech_stack=["Python", "LangChain", "Pinecone", "FastAPI"],
        role="Backend AI Engineer",
    ),
    ProjectRecord(
        project_id="p3",
        title="LedgerIQ - AI Bookkeeping",
        description=(
            "An automated bookkeeping application that uses LLMs to categorize financial transactions "
            "and generate monthly ledger reports with high accuracy."
        ),
        tech_stack=["Node.js", "PostgreSQL", "OpenAI API"],
        role="Full Stack AI Developer",
    ),
]


DUMMY_USER = UserProfile(
    user_id="zain_zia_001",
    full_name="Zain Zia",
    designation="Generative AI Developer & SST (Applied Physics)",
    expertise_areas=[
        "LLMs",
        "RAG systems",
        "Vector Databases",
        "Character Consistency in AI Art",
    ],
    experience_languages=["Node.js", "Python", "React.js"],
    experience_years=5,
    template_type=TemplateType.CUSTOM,
    template_id="custom-template-1",
    selected_template_text=(
        "Hi! I saw your post for [Job Title]. I have extensive experience in [Skills] and recently built "
        "[Project Name] which is very similar to what you need. Let's connect!"
    ),
    custom_template_text=(
        "Hi! I saw your post for [Job Title]. I have extensive experience in [Skills] and recently built "
        "[Project Name] which is very similar to what you need. Let's connect!"
    ),
    notes={"seeded": True},
)


def template_summaries() -> list[TemplateSummary]:
    return [
        TemplateSummary(
            template_id=record.template_id,
            label=record.label,
            description=record.description,
            best_for=record.best_for,
            template_type=record.template_type,
        )
        for record in DEFAULT_TEMPLATES.values()
    ]
