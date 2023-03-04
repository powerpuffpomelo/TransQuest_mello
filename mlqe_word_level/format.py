import pandas as pd
import random


def prepare_data(raw_df, args):
    source_sentences = raw_df["source"].tolist()
    if "target" in raw_df.columns:
        target_sentences = raw_df["target"].tolist()
    elif "pe" in raw_df.columns:
        target_sentences = raw_df["pe"].tolist()
    # source_tags = raw_df["source_tags"].tolist()
    # target_tags = raw_df["target_tags"].tolist()

    sentence_id = 0
    data = []
    if "source_adv_tags" in raw_df.columns and "target_adv_tags" in raw_df.columns:
        source_tags = raw_df["source_tags"].tolist()
        target_tags = raw_df["target_tags"].tolist()
        source_adv_tags = raw_df["source_adv_tags"].tolist()
        target_adv_tags = raw_df["target_adv_tags"].tolist()
        for source_sentence, source_tag_line, target_sentence, target_tag_line, \
                        source_adv_tag_line, target_adv_tag_line in zip(source_sentences, 
                        source_tags, target_sentences, target_tags, source_adv_tags, target_adv_tags):
            for word, tag, adv_tag in zip(source_sentence.split(), source_tag_line.split(), source_adv_tag_line.split()):
                data.append([sentence_id, word, tag, float(adv_tag)])

            data.append([sentence_id, "[SEP]", "OK", 0.5])

            target_words = target_sentence.split()
            target_tags = target_tag_line.split()
            target_adv_tags = target_adv_tag_line.split()

            data.append([sentence_id, args.tag, target_tags.pop(0), float(target_adv_tags.pop(0))])

            for word in target_words:
                data.append([sentence_id, word, target_tags.pop(0), float(target_adv_tags.pop(0))])
                data.append([sentence_id, args.tag, target_tags.pop(0), float(target_adv_tags.pop(0))])

            sentence_id += 1

        return pd.DataFrame(data, columns=['sentence_id', 'words', 'labels', 'adv_labels'])

    elif "source_tags" in raw_df.columns and "target_tags" in raw_df.columns:
        source_tags = raw_df["source_tags"].tolist()
        target_tags = raw_df["target_tags"].tolist()
        for source_sentence, source_tag_line, target_sentence, target_tag_lind in zip(source_sentences, source_tags,
                                                                                    target_sentences, target_tags):
            for word, tag in zip(source_sentence.split(), source_tag_line.split()):
                data.append([sentence_id, word, tag])

            data.append([sentence_id, "[SEP]", "OK"])

            target_words = target_sentence.split()
            target_tags = target_tag_lind.split()

            data.append([sentence_id, args.tag, target_tags.pop(0)])  # gap tag 下划线

            for word in target_words:
                data.append([sentence_id, word, target_tags.pop(0)])
                data.append([sentence_id, args.tag, target_tags.pop(0)])  # gap tag

            sentence_id += 1

        return pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])
    
    else:
        # only parallel
        for source_sentence, target_sentence in zip(source_sentences, target_sentences):
            for word in source_sentence.split():
                data.append([sentence_id, word])

            data.append([sentence_id, "[SEP]"])

            for word in target_sentence.split():
                data.append([sentence_id, word])

            sentence_id += 1

        return pd.DataFrame(data, columns=['sentence_id', 'words'])


def format_to_test(to_test, args):
    test_sentences = []
    for source_sentence, target_sentence in to_test:
        test_sentence = source_sentence + " " + "[SEP]"
        target_words = target_sentence.split()
        for target_word in target_words:
            test_sentence = test_sentence + " " + args.tag + " " + target_word
        test_sentence = test_sentence + " " + args.tag
        test_sentences.append(test_sentence)

    return test_sentences


def post_process(predicted_sentences, test_sentences, args):
    sources_tags = []
    targets_tags = []

    for predicted_sentence, test_sentence in zip(predicted_sentences, test_sentences):
        # print(predicted_sentence) # [{'duplicates': 'BAD'}, {'the': 'OK'}, {'current': 'OK'}, {'set': 'OK'}, {'.': 'OK'}, {'[SEP]': 'OK'}, {'_': 'OK'}, {'Duiert': 'BAD'}, {'_': 'OK'}, {'den': 'OK'}, {'_': 'OK'}, {'aktuellen': 'OK'}, {'_': 'OK'}, {'Satz': 'OK'}, {'_': 'OK'}, {'.': 'OK'}, {'_': 'OK'}]
        # print(test_sentence) # duplicates the current set . [SEP] _ Duiert _ den _ aktuellen _ Satz _ . _
        source_tags = []
        target_tags = []
        words = test_sentence.split()
        is_source_sentence = True
        source_sentence = test_sentence.split("[SEP]")[0]
        target_sentence = test_sentence.split("[SEP]")[1]

        for idx, word in enumerate(words):

            if word == "[SEP]":
                is_source_sentence = False
                continue
            if is_source_sentence:
                if idx < len(predicted_sentence):
                    source_tags.append(list(predicted_sentence[idx].values())[0])
                else:
                    source_tags.append(args.default_quality)
            else:
                if idx < len(predicted_sentence):
                    target_tags.append(list(predicted_sentence[idx].values())[0])
                else:
                    target_tags.append(args.default_quality)

        assert len(source_tags) == len(source_sentence.split())

        if len(target_sentence.split()) > len(target_tags):
            target_tags = target_tags + [args.default_quality for x in
                                         range(len(target_sentence.split()) - len(target_tags))]

        assert len(target_tags) == len(target_sentence.split())
        sources_tags.append(source_tags)
        targets_tags.append(target_tags)

    return sources_tags, targets_tags

def post_process_with_confidence(predicted_sentences, preds_confidences, test_sentences, args):
    sources_tags = []
    targets_tags = []
    sources_confidence = []
    targets_confidence = []

    for predicted_sentence, preds_confidence, test_sentence in zip(predicted_sentences, preds_confidences, test_sentences):
        # print(predicted_sentence) # [{'duplicates': 'BAD'}, {'the': 'OK'}, {'current': 'OK'}, {'set': 'OK'}, {'.': 'OK'}, {'[SEP]': 'OK'}, {'_': 'OK'}, {'Duiert': 'BAD'}, {'_': 'OK'}, {'den': 'OK'}, {'_': 'OK'}, {'aktuellen': 'OK'}, {'_': 'OK'}, {'Satz': 'OK'}, {'_': 'OK'}, {'.': 'OK'}, {'_': 'OK'}]
        # print(test_sentence) # duplicates the current set . [SEP] _ Duiert _ den _ aktuellen _ Satz _ . _
        # assert len(predicted_sentence) == len(preds_confidence)
        # if len(predicted_sentence) != len(test_sentence.split()):
        #     print('---------------------------')
        #     print(len(predicted_sentence))
        #     print(len(test_sentence.split()))
        #     print(predicted_sentence)
        #     print(test_sentence)
        #     assert 1==2
        source_tags = []
        target_tags = []
        source_confidence = []
        target_confidence = []
        words = test_sentence.split()
        is_source_sentence = True
        source_sentence = test_sentence.split("[SEP]")[0]
        target_sentence = test_sentence.split("[SEP]")[1]

        for idx, word in enumerate(words):

            if word == "[SEP]":
                is_source_sentence = False
                continue
            if is_source_sentence:
                if idx < len(predicted_sentence):
                    source_tags.append(list(predicted_sentence[idx].values())[0])
                    source_confidence.append(list(preds_confidence[idx].values())[0])
                else:
                    source_tags.append(args.default_quality)
            else:
                if idx < len(predicted_sentence):
                    target_tags.append(list(predicted_sentence[idx].values())[0])
                    target_confidence.append(list(preds_confidence[idx].values())[0])
                else:
                    target_tags.append(args.default_quality)
                    target_confidence.append(-1)   # 超长

        assert len(source_tags) == len(source_sentence.split())

        if len(target_sentence.split()) > len(target_tags):
            target_tags = target_tags + [args.default_quality for x in
                                         range(len(target_sentence.split()) - len(target_tags))]

        assert len(target_tags) == len(target_sentence.split())
        assert len(target_tags) == len(target_confidence)
        sources_tags.append(source_tags)
        targets_tags.append(target_tags)
        sources_confidence.append(source_confidence)
        targets_confidence.append(target_confidence)

    return sources_tags, targets_tags, sources_confidence, targets_confidence

# def post_process(predicted_sentences, test_sentences):
#     sources_tags = []
#     targets_tags = []
#     for predicted_sentence, test_sentence in zip(predicted_sentences,test_sentences):
#         source_tags = []
#         target_tags = []
#         words = test_sentence.split()
#         source_sentence = True
#         for word_prediction in predicted_sentence:
#             word = list(word_prediction.keys())[0]
#
#             if word == "[SEP]":
#                 source_sentence = False
#                 continue
#             if source_sentence:
#                 source_tags.append(list(word_prediction.values())[0])
#             else:
#                 target_tags.append(list(word_prediction.values())[0])
#         sources_tags.append(source_tags)
#         targets_tags.append(target_tags)
#
#     return sources_tags, targets_tags
