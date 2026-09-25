from qiskit_ibm_runtime import QiskitRuntimeService
from ibm_account import PRIMARY_API_KEY

JOB_ID = "dakokns62pvc739pppe0"

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=PRIMARY_API_KEY,
    instance="open-instance",
)

job = service.job(JOB_ID)

print("Before:", job.status())
job.cancel()
print("After:", job.status())