from enum import Enum
from typing import Dict, Any, Optional
import json
import os
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger("job-service")

# Define job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory storage for job status and results
# Note: In production, this should be replaced with a database
JOB_STORAGE = {}
RESULT_STORAGE = {}

# Storage directory for results
STORAGE_DIR = os.environ.get("STORAGE_DIR", "job_results")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

def update_job_status(
    job_id: str, 
    status: JobStatus, 
    message: str, 
    progress: int = 0
) -> None:
    """
    Update the status of a job in the job storage
    
    Args:
        job_id: Unique identifier for the job
        status: Current status of the job
        message: Status message or description
        progress: Progress percentage (0-100)
    """
    JOB_STORAGE[job_id] = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "progress": progress,
        "updated_at": datetime.now().isoformat()
    }
    logger.info(f"Job {job_id} updated: {status} - {message} ({progress}%)")

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current status of a job
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Dictionary with job status information or None if not found
    """
    status = JOB_STORAGE.get(job_id)
    
    # If job is completed, include the result
    if status and status["status"] == JobStatus.COMPLETED:
        result = get_job_result(job_id)
        if result:
            status["result"] = result
    
    return status

def save_job_result(job_id: str, result: Any) -> None:
    """
    Save the result of a completed job
    
    Args:
        job_id: Unique identifier for the job
        result: The result data to save
    """
    # Store in memory for quick access
    RESULT_STORAGE[job_id] = result
    
    # Serialize to JSON and save to filesystem for persistence
    try:
        result_path = os.path.join(STORAGE_DIR, f"{job_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, default=str)
        logger.info(f"Saved result for job {job_id}")
    except Exception as e:
        logger.error(f"Error saving job result: {str(e)}")

def get_job_result(job_id: str) -> Optional[Any]:
    """
    Get the result of a completed job
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        The job result or None if not found
    """
    # Try to get from memory first
    if job_id in RESULT_STORAGE:
        return RESULT_STORAGE[job_id]
    
    # If not in memory, try to load from filesystem
    try:
        result_path = os.path.join(STORAGE_DIR, f"{job_id}.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result = json.load(f)
                # Cache in memory for future requests
                RESULT_STORAGE[job_id] = result
                return result
    except Exception as e:
        logger.error(f"Error loading job result: {str(e)}")
    
    return None