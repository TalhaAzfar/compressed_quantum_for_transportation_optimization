"""Save the user's IBM credentials in Qiskit's local account store, outside this repo."""
from getpass import getpass


def main():
    from qiskit_ibm_runtime import QiskitRuntimeService

    api_key = getpass('IBM Cloud API key (hidden): ').strip()
    instance_crn = getpass('IBM Quantum instance CRN (hidden): ').strip()
    if not api_key or not instance_crn:
        raise SystemExit('Both an API key and an instance CRN are required.')
    if not instance_crn.lower().startswith('crn:'):
        raise SystemExit('Enter the instance CRN from your IBM Quantum account.')
    QiskitRuntimeService.save_account(
        channel='ibm_quantum_platform',
        token=api_key,
        instance=instance_crn,
        name='transport-aqc',
        overwrite=False,
        set_as_default=False,
    )
    print('Saved account transport-aqc in your local Qiskit account store.')
    print('Credentials are not written into this repository.')
    print('Select it with: export IBM_QUANTUM_ACCOUNT=transport-aqc')


if __name__ == '__main__':
    main()
