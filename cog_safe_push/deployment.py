import requests

from .exceptions import CogSafePushError
from .task_context import TaskContext


def handle_deployment(task_context: TaskContext, version: str) -> None:
    """Create or update a deployment for the model."""
    if not task_context.config.deployment or not task_context.config.deployment.name:
        return

    deployment_name = task_context.config.deployment.name
    deployment_owner = task_context.config.deployment.owner or task_context.model.owner
    deployment_config = task_context.config.deployment

    # Check if deployment exists
    response = requests.get(
        f"https://api.replicate.com/v1/deployments/{deployment_owner}/{deployment_name}",
        headers={"Authorization": f"Bearer {task_context.api_token}"},
    )

    if response.status_code == 404:
        create_deployment(
            task_context, deployment_name, deployment_owner, deployment_config, version
        )
    else:
        update_deployment(
            task_context, deployment_name, deployment_owner, response.json(), version
        )


def create_deployment(
    task_context: TaskContext,
    deployment_name: str,
    deployment_owner: str,
    deployment_config,
    version: str,
) -> None:
    """Create a new deployment for the model."""
    hardware = deployment_config.hardware or "cpu"
    if hardware == "cpu":
        min_instances = 1
        max_instances = 20
    else:
        min_instances = 0
        max_instances = 2

    print(
        f"Creating deployment {deployment_name} with {hardware} hardware, {min_instances} min instances, {max_instances} max instances"
    )

    # Create new deployment
    response = requests.post(
        "https://api.replicate.com/v1/deployments",
        headers={
            "Authorization": f"Bearer {task_context.api_token}",
            "Content-Type": "application/json",
        },
        json={
            "name": deployment_name,
            "owner": deployment_owner,
            "model": f"{task_context.model.owner}/{task_context.model.name}",
            "version": version,
            "hardware": hardware,
            "min_instances": min_instances,
            "max_instances": max_instances,
        },
    )

    if response.status_code not in (200, 201):
        raise CogSafePushError(f"Failed to create deployment: {response.text}")


def update_deployment(
    task_context: TaskContext,
    deployment_name: str,
    deployment_owner: str,
    current_deployment: dict,
    version: str,
) -> None:
    """Update an existing deployment for the model."""
    print(
        f"Updating deployment {deployment_name} with {current_deployment['hardware']} hardware, {current_deployment['min_instances']} min instances, {current_deployment['max_instances']} max instances"
    )
    print(f"Changing version from {current_deployment['version']} to {version}")

    response = requests.patch(
        f"https://api.replicate.com/v1/deployments/{deployment_owner}/{deployment_name}",
        headers={
            "Authorization": f"Bearer {task_context.api_token}",
            "Content-Type": "application/json",
        },
        json={
            "version": version,
        },
    )

    if response.status_code not in (200, 201):
        raise CogSafePushError(f"Failed to update deployment: {response.text}")
