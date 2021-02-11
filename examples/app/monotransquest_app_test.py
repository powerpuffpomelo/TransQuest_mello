import logging

from transquest.app.monotransquest_app import MonoTransQuestApp


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

test_sentences = [
    [
        "Jocurile de oferă noi provocări pentru IA în domeniul teoriei jocurilor.",
        "Games provide new challenges for IA in the area of gambling theory"
    ]
]

app = MonoTransQuestApp("monotransquest-da-si_en", use_cuda=False, force_download=True)
print(app.predict_quality(test_sentences))
