"""Helpers for adding local recovery guidance to generated contexts."""

from src.decoding.failures import FailureReason


def inject_instruction(context: str, reason: FailureReason) -> str:
    """Insert concise, reason-specific guidance into the first system header.

    Args:
        context: A complete ChatML context.
        reason: Typed failure reason that prompted the retry.

    Returns:
        Context with guidance prepended to its system message.

    Raises:
        ValueError: If the context does not start with the supported header.
    """
    header = "<|im_start|>system\n"
    if not context.startswith(header):
        raise ValueError("Recovery requires a ChatML system header.")
    guidance = {
        "token_limit": (
            "Return only the requested input value and finish its JSON "
            "boundary. Omit explanations; preserve all requested content."
        ),
        "no_continuation": (
            "Provide a complete value directly; avoid an unfinished prefix."
        ),
        "nonfinite_number": "Give a finite JSON number.",
        "invalid_value": (
            "Give one valid value matching the declared JSON type."
        ),
        "byte_limit": "Give a shorter value.",
        "context_limit": "Give a concise value.",
        "invalid_logits": "Give a valid value.",
    }[reason]
    return header + guidance + "\n" + context[len(header):]
