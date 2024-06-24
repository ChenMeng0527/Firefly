'''
{"conversation_id": 1,
 "category": "Brainstorming",
 "conversation": [{"human": "如何保障工作中遵循正确的安全准则？",
                   "assistant": "为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。"},
                  {"human": "如何应对突发紧急情况？",
                   "assistant": "突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。"},
                  ],
 "dataset": "moss"}


'''

import os
data_from = {"conversation_id": 1,
             "category": "Brainstorming",
             "conversation": [{"human": "HUMAN_TEXT",
                               "assistant": "ASSITANT_TEXT"},
                              ],
             "dataset": "moss"}

data_path = '/home/hadoop/ChenmNLP/tasks/OwnKnowledgeBase/data/question_from_wengao_detail/'
id = ['2924', '3485']

def get_qa_from_text(file_name):

    query_answer = {}
    data_all = []
    with open(file_name, 'r') as f:
        data = f.readlines()
        for i in data:
            if i == '\n':
                pass
            else:
                temp_ = i.replace('\n', '')
                data_all.append(temp_)

    index = 0
    len_max = len(data_all)
    while index < len_max:
        target_ = data_all[index]
        # data_all[52364]
        if 'Q:' in target_:
            q = target_.split('Q:')[1]

            # 如果出现过历史问题，则跳过
            if query_answer.get(q):
                pass
            else:
                a = data_all[index + 1].split('A:')[1]
                query_answer[q] = a
            index += 5
        else:
            index += 1
    return query_answer


# ---获取数据形成qa----
query_answer_all = {}
for i in id:
    file_dir = data_path + 'plan_' + i
    # 读取下面所有文件
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        print(i, file_name)
        file_path = file_dir + '/' + file_name
        query_answer = get_qa_from_text(file_path)
        query_answer_all.update(query_answer)
        print('len_query_answer_all', len(query_answer_all))


# ---将qa数据写入到固定格式---
data_from = {"conversation_id": 1,
             "category": "Brainstorming",
             "conversation": [],
             "dataset": "moss"}
data_all = []
for q, a in list(query_answer_all.items())[0:1000]:
    data_from = {"conversation_id": 1,
                 "category": "Brainstorming",
                 "conversation": [],
                 "dataset": "moss"}
    temp_key = {"human": q, "assistant": a}
    data_from['conversation'].append(temp_key)
    print(data_from)
    data_all.append(data_from)



# ---将数据写入到aaa.jsonl文件中---
import json
with open('/home/hadoop/Firefly/data/youshu_aq_data.json', 'w', encoding='utf-8') as f:
    for entry in data_all:
        json_line = json.dumps(entry, ensure_ascii=False)
        f.write(json_line + "\n")

