"""
notify.py -- optional Telegram progress reporting (progress.md S.40.5/41.3).

Reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from the environment if set,
otherwise sends nothing. urllib only (no dependency).
Tries a normal (cert-verified) HTTPS POST first; if that fails (common Windows
SSL cert issue), retries with an unverified SSL context. Never raises during a
run; run `python notify.py "msg"` to see the actual error if it won't send.
"""
import os
import ssl
import urllib.request
import urllib.parse

_DEFAULT_TOKEN = ""
_DEFAULT_CHAT = ""


def notify(msg, verbose=False):
    tok = os.environ.get('TELEGRAM_BOT_TOKEN') or _DEFAULT_TOKEN
    cid = os.environ.get('TELEGRAM_CHAT_ID') or _DEFAULT_CHAT
    if not tok or not cid:
        return False
    data = urllib.parse.urlencode({'chat_id': cid, 'text': msg}).encode()
    url = f'https://api.telegram.org/bot{tok}/sendMessage'
    last = None
    for ctx in (None, ssl._create_unverified_context()):
        try:
            urllib.request.urlopen(url, data=data, timeout=15, context=ctx)
            return True
        except Exception as e:
            last = e
    if verbose:
        print('notify error:', repr(last))
    return False


if __name__ == '__main__':
    import sys
    ok = notify(' '.join(sys.argv[1:]) or 'notify.py test message', verbose=True)
    print('sent' if ok else 'NOT sent')
