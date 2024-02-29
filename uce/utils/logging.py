import logging
from typing import Optional, TextIO


def setup_logger(
    name: Optional[str] = 'uce',
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = '%(asctime)s %(name)s %(levelname)s: %(message)s',
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
) -> logging.Logger:
    """Setups logger: name, level, format etc.

    Code is adapted from https://pytorch.org/ignite/_modules/ignite/utils.html#setup_logger # noqa

    Args:
        name: new name for the logger. If None, the standard logger is used.
        level: logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG.
        stream: logging stream. If None, the standard stream is used (sys.stderr).
        format: logging format.
        filepath: Optional logging file path. If not None, logs are written to the file.
        distributed_rank: Optional, rank in distributed configuration to avoid
            logger setup for workers. If None, distributed_rank is initialized to
            the rank of process.
        reset: if True, reset an existing logger rather than keep format, handlers, and level.

    Returns:
        logging.Logger
    """
    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import torch.distributed as dist
        if dist.is_initialized():
            distributed_rank = dist.get_rank()
        else:
            distributed_rank = 0

    # Remove previous handlers
    if distributed_rank > 0 or reset:
        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:
        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger
