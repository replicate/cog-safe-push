from unittest.mock import MagicMock, patch

import pytest

from cog_safe_push.deployment import (
    create_deployment,
    handle_deployment,
    update_deployment,
)
from cog_safe_push.exceptions import CogSafePushError
from cog_safe_push.task_context import TaskContext


@pytest.fixture
def mock_replicate():
    with patch("replicate.deployments") as mock:
        mock.get = MagicMock()
        mock.create = MagicMock()
        mock.update = MagicMock()
        yield mock


@pytest.fixture
def task_context():
    context = MagicMock(spec=TaskContext)
    context.model = MagicMock()
    context.model.owner = "test-owner"
    context.model.name = "test-model"
    context.deployment_name = "test-deployment"
    context.deployment_owner = "test-owner"
    context.deployment_hardware = "cpu"
    return context


def test_no_deployment_config(task_context, mock_replicate):
    task_context.deployment_name = None
    handle_deployment(task_context, "test-version")
    mock_replicate.get.assert_not_called()
    mock_replicate.create.assert_not_called()


def test_create_deployment(task_context, mock_replicate):
    mock_replicate.get.side_effect = Exception("not found")
    handle_deployment(task_context, "test-version")
    mock_replicate.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_create_deployment_error(task_context, mock_replicate):
    mock_replicate.get.side_effect = Exception("not found")
    mock_replicate.create.side_effect = Exception("create failed")
    with pytest.raises(CogSafePushError, match="Failed to create deployment"):
        handle_deployment(task_context, "test-version")


def test_update_deployment(task_context, mock_replicate):
    current_deployment = MagicMock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20
    mock_replicate.get.return_value = current_deployment

    handle_deployment(task_context, "test-version")
    mock_replicate.update.assert_called_once_with(
        deployment_owner="test-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_update_deployment_error(task_context, mock_replicate):
    current_deployment = MagicMock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20
    mock_replicate.get.return_value = current_deployment
    mock_replicate.update.side_effect = Exception("update failed")

    with pytest.raises(CogSafePushError, match="Failed to update deployment"):
        handle_deployment(task_context, "test-version")


def test_update_deployment_different_owners(task_context, mock_replicate):
    current_deployment = MagicMock()
    current_deployment.owner = "different-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20
    mock_replicate.get.return_value = current_deployment

    handle_deployment(task_context, "test-version")
    mock_replicate.update.assert_called_once_with(
        deployment_owner="different-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_update_deployment_function(mock_replicate):
    """Test the update_deployment function directly."""
    current_deployment = MagicMock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20

    update_deployment(current_deployment, "test-version")
    mock_replicate.update.assert_called_once_with(
        deployment_owner="test-owner",
        deployment_name="test-deployment",
        version="test-version",
    )


def test_update_deployment_function_error(mock_replicate):
    """Test error handling in update_deployment function."""
    current_deployment = MagicMock()
    current_deployment.owner = "test-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20
    mock_replicate.update.side_effect = Exception("update failed")

    with pytest.raises(CogSafePushError, match="Failed to update deployment"):
        update_deployment(current_deployment, "test-version")


def test_create_deployment_no_name(task_context):
    task_context.deployment_name = None
    with pytest.raises(CogSafePushError, match="Deployment name is required"):
        create_deployment(task_context, "test-version")


def test_create_deployment_cpu(task_context, mock_replicate):
    mock_replicate.get.side_effect = Exception("not found")
    handle_deployment(task_context, "test-version")
    mock_replicate.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_create_deployment_gpu(task_context, mock_replicate):
    task_context.deployment_hardware = "gpu-t4"
    mock_replicate.get.side_effect = Exception("not found")
    handle_deployment(task_context, "test-version")
    mock_replicate.create.assert_called_once_with(
        name="test-deployment",
        model="test-owner/test-model",
        version="test-version",
        hardware="gpu-t4",
        min_instances=0,
        max_instances=2,
    )


def test_handle_deployment_different_owners(task_context, mock_replicate):
    task_context.model.owner = "model-owner"
    task_context.deployment_owner = "deployment-owner"
    mock_replicate.get.side_effect = Exception("not found")
    handle_deployment(task_context, "test-version")
    mock_replicate.create.assert_called_once_with(
        name="test-deployment",
        model="model-owner/test-model",
        version="test-version",
        hardware="cpu",
        min_instances=1,
        max_instances=20,
    )


def test_handle_deployment_update_different_owners(task_context, mock_replicate):
    task_context.model.owner = "model-owner"
    task_context.deployment_owner = "deployment-owner"
    current_deployment = MagicMock()
    current_deployment.owner = "deployment-owner"
    current_deployment.name = "test-deployment"
    current_deployment.current_release.version = "old-version"
    current_deployment.current_release.configuration.hardware = "cpu"
    current_deployment.current_release.configuration.min_instances = 1
    current_deployment.current_release.configuration.max_instances = 20
    mock_replicate.get.return_value = current_deployment

    handle_deployment(task_context, "test-version")
    mock_replicate.update.assert_called_once_with(
        deployment_owner="deployment-owner",
        deployment_name="test-deployment",
        version="test-version",
    )
