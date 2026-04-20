from __future__ import annotations

import os
import re
from pypdf import PdfReader

from app.schemas import ProjectRecord
from app.vector_store import get_project_store


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"
    return full_text


def chunk_projects(text: str) -> list[ProjectRecord]:
    projects: list[ProjectRecord] = []
    project_pattern = r"(?=Project\s+\d+:)"
    parts = re.split(project_pattern, text)

    for index, part in enumerate(parts, start=1):
        part = part.strip()
        if not part.startswith("Project") or len(part) < 50:
            continue

        clean_content = re.sub(r"\n\s*\n", "\n", part)
        lines = [line.strip() for line in clean_content.split("\n") if line.strip()]
        title = lines[0] if lines else f"Imported Project {index}"

        tech_stack_match = next((line for line in lines if line.lower().startswith("tech")), "")
        tech_stack = tech_stack_match.split(":", 1)[1].strip() if ":" in tech_stack_match else ""

        projects.append(
            ProjectRecord(
                project_id=f"imported_{index}",
                title=title,
                description=clean_content,
                tech_stack=tech_stack,
                role="Imported Portfolio Project",
            )
        )

    return projects


def seed_database(file_path: str, user_id: str) -> list[ProjectRecord]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' was not found.")

    text = extract_text_from_pdf(file_path)
    projects = chunk_projects(text)
    get_project_store().upsert_projects(user_id=user_id, projects=projects)
    return projects


if __name__ == "__main__":
    imported = seed_database("Project_Details.pdf", user_id="imported_user")
    print(f"Imported {len(imported)} projects.")
