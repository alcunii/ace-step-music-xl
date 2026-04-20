"""Tests for the GitHub Actions workflow structure."""
import yaml
import pytest
from pathlib import Path

WORKFLOW_PATH = Path(__file__).parent / ".github" / "workflows" / "deploy.yml"


@pytest.fixture
def workflow():
    with open(WORKFLOW_PATH) as f:
        return yaml.safe_load(f)


class TestWorkflowStructure:
    def test_workflow_file_exists(self):
        assert WORKFLOW_PATH.exists(), f"Workflow file not found at {WORKFLOW_PATH}"

    def test_triggers_on_push_to_main(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        assert "main" in triggers["push"]["branches"]

    def test_triggers_on_pr_to_main(self, workflow):
        triggers = workflow.get("on") or workflow.get(True)
        assert "main" in triggers["pull_request"]["branches"]

    def test_has_three_jobs(self, workflow):
        jobs = workflow["jobs"]
        assert set(jobs.keys()) == {"test", "build-push", "deploy"}

    def test_build_push_needs_test(self, workflow):
        assert workflow["jobs"]["build-push"]["needs"] == "test"

    def test_deploy_needs_build_push(self, workflow):
        assert workflow["jobs"]["deploy"]["needs"] == "build-push"

    def test_build_push_only_on_main(self, workflow):
        condition = workflow["jobs"]["build-push"]["if"]
        assert "push" in condition
        assert "main" in condition

    def test_deploy_only_on_main(self, workflow):
        condition = workflow["jobs"]["deploy"]["if"]
        assert "push" in condition
        assert "main" in condition

    def test_build_uses_linux_amd64(self, workflow):
        build_steps = workflow["jobs"]["build-push"]["steps"]
        build_step = [s for s in build_steps if s.get("name") == "Build and push"][0]
        assert build_step["with"]["platforms"] == "linux/amd64"

    def test_image_name_is_xl(self, workflow):
        steps = workflow["jobs"]["build-push"]["steps"]
        meta = [s for s in steps if s.get("id") == "meta"][0]
        assert meta["with"]["images"] == "dmrabh/ace-step-music-xl"

    def test_deploy_references_xl_image(self, workflow):
        """Deploy job announces which image was pushed. Template update via
        RunPod UI — GraphQL saveTemplate mutation no longer usable with the
        minimal payload (now requires name/env/containerDiskInGb/etc.)."""
        deploy_steps = workflow["jobs"]["deploy"]["steps"]
        deploy_step = deploy_steps[0]
        assert "IMAGE_TAG" in deploy_step["env"]
        assert "dmrabh/ace-step-music-xl" in deploy_step["env"]["IMAGE_TAG"]
