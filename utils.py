from functools import wraps
from threading import Thread
import time

def memoize(function):
    """Provides a decorator for memoizing functions"""
    memo = {}
    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper


def setInterval (callback, interval):
    cancelled = False

    def timerExecutor ():
        nonlocal cancelled

        nextCallTimestamp = time.time()
        while cancelled != True:
            nextCallTimestamp = nextCallTimestamp + interval
            # try:
            callback()
            # except Exception as e:
            #     print(f'Exception while executing interval callback: {e}')
            time.sleep(max(0, nextCallTimestamp - time.time()))

    def cancelTimer ():
        nonlocal cancelled
        cancelled = True

    timerThread = Thread(target=timerExecutor, daemon=False)
    timerThread.start()
    return cancelTimer


def runAsync (callback):
    thread = Thread(target=callback, daemon=False)
    thread.start()
