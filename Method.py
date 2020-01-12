import copy
import json

from konlpy.tag import Okt
from sklearn.linear_model import LogisticRegression

def Normalize(l):
    # list의 원소들의 값을 normalize
    temp = sum([i ** 2 for i in l]) ** (1/2)
    if temp == 0:
        return l
    else:
        return [i / temp for i in l]

def MakeFeature(chats, feature_name, feature_name_list, T):
    okt = Okt()
    name = ''
    con = 0
    feature = copy.deepcopy(feature_name)
    for chat in chats:
        # 각 값은 feature_name에 매치됨
        if con == 1:
            # 공백일 때, 대화를 이어질 수도 있는 경우가 있으므로
            # 한번 더 체크
            if ('년' in chat) & ('월' in chat) & ('일' in chat) &\
            ('오' in chat) & (':' in chat):
                if ',' in chat:
                    pass
                else:
                    con = 0
                    name = ''
        if len(chat.split()) == 0:
            # 공백일 때, 날짜별로 분류하는 경우이므로
            # name을 초기화 하고 스킵
            con = 1
            continue
        try:
            temp = chat.split(' : ')
            if len(temp) <= 1:
                # 하나의 대화를 \n으로 분류한 경우
                chat1 = chat
            else:
                # 정상적인 대화
                temp1 = temp[0].split(', ')
                name = temp1[1]
                chat1 = temp[1]
            if name == '':
                # name이 없을 때, 스킵
                continue
            if chat1 == '사진':
                # 사진 보낸 경우
                feature['@사진'] += 1
            elif chat1 == '이모티콘':
                # 이모티콘 보낸 경우
                feature['@이모티콘'] += 1
            elif len(chat1) <= 4:
                # 파일 보낸 경우를 카운트하기 위해 예외처리
                for k, v in okt.pos(chat1, norm = True, stem = True):
                    try:
                        if T == 1:
                            feature.setdefault(k, 0)
                            feature[k] += 1
                            feature_name.setdefault(k, 0)
                            if k not in feature_name_list:
                                feature_name_list.append(k)
                        if T == 2:
                            if k in feature:
                                feature[k] += 1
                    except:
                        pass
            else:
                if '파일:' in chat1:
                    # 파일 보낸 경우
                    feature['@파일'] += 1
                else:
                    # 일반적인 대화
                    for k, v in okt.pos(chat1, norm = True, stem = True):
                        try:
                            if T == 1:
                                feature.setdefault(k, 0)
                                feature[k] += 1
                                feature_name.setdefault(k, 0)
                                if k not in feature_name_list:
                                    feature_name_list.append(k)
                            if T == 2:
                                if k in feature:
                                    feature[k] += 1
                        except:
                            pass
        except:
            pass
    feature_list = []
    for i in feature_name_list:
        feature_list.append(feature[i])
    if T == 1:
        return Normalize(feature_list), feature_name, feature_name_list
    elif T == 2:
        return Normalize(feature_list)

def Similarity(feature, feature_set, label_set):
    model = LogisticRegression(fit_interceptbool = False)
    model.fit(feature_set, label_set)
    result = model.predict_proba([feature])
    return result[0][1]

def Main(chats, T, label = None):
    try:
        with open('feature.json', 'r', encoding = 'utf-8') as f:
            feature_name = json.load(f)
        feature_name_list = feature_name['@리스트']
        feature_set = feature_name['@피처']
        label_set = feature_name['@라벨']
    except:
        feature_name = {'@사진': 0, '@이모티콘': 0, '@파일': 0}
        feature_name_list = ['@사진', '@이모티콘', '@파일']
        feature_set = []
        label_set = []
    if T == 1:
        # T = 1: 학습할 데이터에서 feature로 뽑기, label이 존재
        try:
            feature, feature_name, feature_name_list =\
            MakeFeature(chats, feature_name, feature_name_list, T)
            feature_set.append(feature)
            M = max([len(i) for i in feature_set])
            for i in range(len(feature_set)):
                feature_set[i] = feature_set[i] + [0] * (M - len(feature_set[i]))
            label_set.append(label)
            feature_name['@리스트'] = feature_name_list
            feature_name['@피처'] = feature_set
            feature_name['@라벨'] = label_set
            with open('feature.json', 'w', encoding = 'utf-8') as f1:
                json.dump(feature_name, f1, ensure_ascii=False, indent = '\t')
            return 'Finish'
        except:
            return 'Failure'
    elif T == 2:
        # T = 2: 결과를 보내줄 데이터에서 feature를 뽑기
        feature = MakeFeature(chats, feature_name, feature_name_list, T)
        result = Similarity(feature, feature_set, label_set)
        return result

