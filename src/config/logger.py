import logging.config
import time

from starlette.middleware.base import BaseHTTPMiddleware


class RouterLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
    self,
    app,
    *,
    logger
    ) -> None:
        self._logger = logger
        super().__init__(app)

    async def dispatch(self,request,call_next):

        try:
            start_time = time.perf_counter()
            response = await self._execute_request(call_next, request)
            finish_time = time.perf_counter()

            execution_time = finish_time - start_time

            self._logger.info(f'{request.client.host}:{request.client.port} :: {request.method} :: {request.url.path} :: {execution_time:0.4f}s')
        except:
            print('error occured')
        return response


    async def _execute_request(self,
    call_next,
    request
    ):
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            self._logger.error(f'error occured while processing the {request.method} request {request.url.path}: {e}')

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
            'standard': {
            'format': '%(asctime)s::%(levelname)s::%(name)s::%(lineno)d:: %(message)s'
            },
                },
    'handlers': {
            'console': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout', # Default is stderr
                    },
            'file': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.FileHandler',
                    'filename': 'logs.log'
                    },
            },
    'loggers': {
            '': { # root logger
                'handlers': ['console','file'],
                'level': 'DEBUG',
                'propogate' : 'False'
            }
        }
    }

def setup_logging(config_dict=LOGGING_CONFIG):
    """
    The setup_logging function reads a YAML file containing logging configuration information.
    The function then uses the logging module's config.dictConfig() method to configure the logger.

    :param config_file: Pass in the name of a yaml file that contains logging configuration information
    :return: Nothing
    :doc-author: Trelent
    """
    # with open('./logging_config.yml', 'rt') as f:
    # config = yaml.safe_load(f.read())

    logging.config.dictConfig(config_dict)
    # print('logging setup done ')


def get_logger(name: str = None):
    """
    The get_logger function is a wrapper for the logging module's getLogger function.
    It returns a logger object that can be used to log messages at different levels of severity.
    The default level is WARNING, but this can be changed by calling setLevel on the returned logger object.

    :param name: str: Set the name of the logger
    :return: The logger object
    :doc-author: Trelent
    """
    setup_logging()
    return logging.getLogger(name)