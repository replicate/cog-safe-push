import pytest
from unittest.mock import Mock

from cog_safe_push.config import Config, DeploymentConfig
from cog_safe_push.deployment import (
    handle_deployment,
    create_deployment,
    update_deployment,
)
from cog_safe_push.exceptions import CogSafePushError
from cog_safe_push.task_context import TaskContext


@pytest.fixture
def task_context():
    context = Mock(spec=TaskContext)
    context.model = Mock()
    context.model.owner = "test-owner"
    context.model.name = "test-model"
    context.client = Mock()
    context.deployment_name = None
    context.deployment_owner = None
    context.deployment_hardware = None
    return context


def test_handle_deployment_no_config(task_context):
    """Test that handle_deployment does nothing when no deployment config exists."""
    handle_deployment(task_context, "test-version")
    task_context.client.deployments.get.assert_not_called()


def test_handle_deployment_no_name(task_context):
    """Test that handle_deployment does nothing when deployment has no name."""
    task_context.deployment_name = None
    handle_deployment(task_context, "test-version")
    task_context.client.deployments.get.assert_not_called()


def test_handle_deployment_create_new(task_context):
    """Test creating a new deployment."""
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.deployment_hardware = "cpu"
    task_context.client.deployments.get.side_effect = Exception("not found")

    handle_deployment(task_context, "test-version")

    task_context.client.deployments.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_handle_deployment_update_existing(task_context):
    """Test updating an existing deployment."""
    current_deployment = Mock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release = Mock()
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration = Mock()
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.deployment_hardware = "cpu"
    task_context.client.deployments.get.return_value = current_deployment

    handle_deployment(task_context, "test-version")

    task_context.client.deployments.update.assert_called_once_with(
        deployment_owner="test-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_handle_deployment_error(task_context):
    """Test handling of unexpected errors."""
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.client.deployments.get.side_effect = Exception("unexpected error")

    with pytest.raises(CogSafePushError, match="Failed to check deployment"):
        handle_deployment(task_context, "test-version")


def test_create_deployment_cpu(task_context):
    """Test creating a CPU deployment."""
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.deployment_hardware = "cpu"

    create_deployment(task_context, "test-version")

    task_context.client.deployments.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_create_deployment_gpu(task_context):
    """Test creating a GPU deployment."""
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.deployment_hardware = "gpu-t4"

    create_deployment(task_context, "test-version")

    task_context.client.deployments.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="gpu-t4",
        min_instances=0,
        max_instances=2,
    )


def test_create_deployment_error(task_context):
    """Test handling of deployment creation errors."""
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "test-owner"
    task_context.client.deployments.create.side_effect = Exception("creation error")

    with pytest.raises(CogSafePushError, match="Failed to create deployment"):
        create_deployment(task_context, "test-version")


def test_update_deployment(task_context):
    """Test updating an existing deployment."""
    current_deployment = Mock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release = Mock()
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration = Mock()
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    update_deployment(task_context, current_deployment, "test-version")

    task_context.client.deployments.update.assert_called_once_with(
        deployment_owner="test-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_update_deployment_error(task_context):
    """Test handling of deployment update errors."""
    current_deployment = Mock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release = Mock()
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration = Mock()
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    task_context.client.deployments.update.side_effect = Exception("update error")

    with pytest.raises(CogSafePushError, match="Failed to update deployment"):
        update_deployment(task_context, current_deployment, "test-version")


def test_handle_deployment_different_owners(task_context):
    """Test creating a deployment when model owner differs from deployment owner."""
    task_context.model.owner = "model-owner"
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "deployment-owner"
    task_context.deployment_hardware = "cpu"
    task_context.client.deployments.get.side_effect = Exception("not found")

    handle_deployment(task_context, "test-version")

    task_context.client.deployments.create.assert_called_once_with(
        name="test-deployment",
        model="model-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_handle_deployment_update_different_owners(task_context):
    """Test updating a deployment when model owner differs from deployment owner."""
    task_context.model.owner = "model-owner"
    current_deployment = Mock()
    current_deployment.owner = "deployment-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release = Mock()
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration = Mock()
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "deployment-owner"
    task_context.deployment_hardware = "cpu"
    task_context.client.deployments.get.return_value = current_deployment

    handle_deployment(task_context, "test-version")

    task_context.client.deployments.update.assert_called_once_with(
        deployment_owner="deployment-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_create_deployment_different_owners(task_context):
    """Test creating a deployment when model owner differs from deployment owner."""
    task_context.model.owner = "model-owner"
    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "deployment-owner"
    task_context.deployment_hardware = "cpu"

    create_deployment(task_context, "test-version")

    task_context.client.deployments.create.assert_called_once_with(
        name="test-deployment",
        model="model-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_update_deployment_different_owners(task_context):
    """Test updating a deployment when model owner differs from deployment owner."""
    task_context.model.owner = "model-owner"
    current_deployment = Mock()
    current_deployment.owner = "deployment-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release = Mock()
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration = Mock()
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    task_context.deployment_name = "test-deployment"
    task_context.deployment_owner = "deployment-owner"
    task_context.deployment_hardware = "cpu"

    update_deployment(task_context, current_deployment, "test-version")

    task_context.client.deployments.update.assert_called_once_with(
        deployment_owner="deployment-owner",
        deployment_name="test-deployment",
        version="test-version",
    )
