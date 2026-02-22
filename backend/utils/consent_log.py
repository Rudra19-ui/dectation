from models.consent_log import log_consent as _log_consent

class _ConsentLogger:
    def log_consent(self, request_id: str, consent: bool, client_host: str):
        return _log_consent(consent, client_host, request_id=request_id)

consent_logger = _ConsentLogger()