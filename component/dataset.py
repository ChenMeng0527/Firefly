import json
from loguru import logger
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    '''
    pytorch dataset模块, ！！三要素！！
        1：init
        2:__len__
        3:__getitem__
    '''
    def __init__(self, file, tokenizer, max_seq_length):
        '''

        '''
        # tokenizer
        self.tokenizer = tokenizer
        # bos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        # eos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        # 最大长度
        self.max_seq_length = max_seq_length

        # 读取文件数据
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):

        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        # 收集多轮对话
        utterances = []
        for x in conversation:
            utterances.append(x['human'])
            utterances.append(x['assistant'])
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
        input_ids = [self.bos_token_id]

        # 重点：用于对input进行mask，只计算target部分的loss
        target_mask = [0]
        for i, utterances_id in enumerate(utterances_ids):
            # 每一个q/a后面加上</s>
            input_ids += (utterances_id + [self.eos_token_id])
            # 如果是query, target_mask为0
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id) + 1)
            # 如果是answer, target_mask为1，注意target1</s>都为1
            else:
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)

        # 对长度进行截断
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs




class ChatGLM2SFTDataset(SFTDataset):

    def __getitem__(self, index):
        """
        基本沿袭ChatGLM2的指令微调的格式，做了小修改，多轮对话如下。
        ！！！index后为一段完整的对话！！！
        生成'input_ids' / 'attention_mask' / 'target_mask'这三个
        #                  问 问 问  答 答 答  停止符
        # input_ids:      [1, 2, 3, 3, 4, 5, 'stop', 2, 3, 4, 5, 6, 7, 'stop']
        # target_mask:    [0, 0, 0, 1, 1, 1, 1,      0, 0, 0, 1, 1, 1, 1]
        # attention_mask: [1, 1, 1, 1, 1, 1, 1,      1, 1, 1, 1, 1, 1, 1]
        """
        # {"conversation_id": 1,
        #  "category": "Brainstorming",
        #  "conversation": [{"human": "如何保障工作中遵循正确的安全准则？",
        #                    "assistant": "为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。"},
        #                   {"human": "如何应对突发紧急情况？",
        #                    "assistant": "突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。"},
        #                   {"human": "在网络安全方面，如何避免遭受黑客攻击？",
        #                    "assistant": "以下是一些避免遭受黑客攻击的建议：\n\n1.更新安全软件：确保您的操作系统、应用程序和安全软件是最新的，并随时更新它们。这可以帮助您及时发现和修复安全漏洞。\n\n2.使用强密码：使用强密码，包括字母、数字和特殊字符。不要使用容易猜测的密码，如生日、姓名或常用单词。\n\n3.使用多因素身份验证：使用多因素身份验证来增加账户的安全性，例如发送短信验证码或使用指纹识别。\n\n4.小心附件和链接：不要打开未知的附件或链接，因为它们可能包含恶意软件。\n\n5.备份重要文件：定期备份重要文件，并将其存储在安全的位置。\n\n6.保护个人信息：不要在互联网上公开个人信息，例如出生日期、地址或社交安全号码。\n\n7.使用加密连接：在访问敏感信息时，使用加密连接，如HTTPS。\n\n8.教育员工：向员工提供网络安全培训，并确保他们了解正确的安全措施和行为。\n\n总之，避免遭受黑客攻击需要采取多种措施，包括更新安全软件、使用强密码、使用多因素身份验证、小心附件和链接、备份重要文件、保护个人信息、使用加密连接以及教育员工。这些措施可以帮助您保护个人信息和数据免受黑客攻击。"},
        #                   {"human": "如何确保网络连接的安全性？",
        #                    "assistant": "以下是一些确保网络连接安全性的建议：\n\n1.使用加密连接：使用加密连接，如HTTPS，以确保数据传输过程中不被窃听或篡改。\n\n2.使用防病毒软件：安装和更新防病毒软件，可以帮助检测和清除恶意软件。\n\n3.使用防火墙：使用防火墙保护网络和设备，可以帮助阻止未经授权的访问和攻击。\n\n4.禁止未经授权的访问：禁止未经授权的访问，包括限制网络访问和设备访问权限。\n\n5.定期更新操作系统和应用程序：定期更新操作系统和应用程序，以修复安全漏洞和缺陷。\n\n6.使用强密码：使用强密码，包括字母、数字和特殊字符，并定期更改密码。\n\n7.控制访问：控制访问网络和设备的人员，仅授权人员可访问重要资源。\n\n8.教育员工：向员工提供网络安全培训，并确保他们了解正确的安全措施和行为。\n\n总之，确保网络连接的安全性需要采取多种措施，包括使用加密连接、使用防病毒软件、使用防火墙、禁止未经授权的访问、定期更新操作系统和应用程序、使用强密码、控制访问和教育员工。这些措施可以帮助您保护网络和设备免受未经授权的访问和攻击。"},
        #                   {"human": "对于金融机构来说，如何保护客户的隐私和数据安全？",
        #                    "assistant": "金融机构需要对客户的隐私和数据安全高度重视，以下是一些保护客户隐私和数据安全的建议：\n\n1.遵守合规要求：确保遵守所有适用的合规要求和监管标准，包括数据隐私法规和安全标准。\n\n2.加强身份验证：采用多因素身份验证，例如使用短信验证码、指纹识别等，以确保客户身份的真实性。\n\n3.加密敏感数据：使用加密技术对敏感数据进行加密，以保护客户信息不被未经授权的人员获得。\n\n4.建立安全控制：建立安全控制，包括访问控制、安全审计和安全事件响应，以保护客户信息的安全性。\n\n5.教育员工：向员工提供网络安全培训，并确保他们了解正确的安全措施和行为。\n\n6.定期测试：定期进行安全测试和漏洞扫描，以发现和修复潜在的安全漏洞。\n\n7.建立应急计划：建立应急计划，并对其进行定期测试，以确保在紧急情况下能够及时响应和恢复。\n\n总之，保护客户隐私和数据安全需要采取多种措施，包括遵守合规要求、加强身份验证、加密敏感数据、建立安全控制、教育员工、定期测试和建立应急计划。这些措施可以帮助金融机构保护客户信息的安全性和隐私。"}],
        #  "dataset": "moss"}

        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']
        input_format = '[Round {}]\n\n问：{}\n\n答：'
        target_format = '{}'

        # 收集多轮对话
        utterances = []
        # x: [{"human": "如何保障工作中遵循正确的安全准则？",
        #      "assistant": "为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。"},
        #     {"human": "如何应对突发紧急情况？",
        #      "assistant": "突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。"},


        for i, x in enumerate(conversation):
            human = input_format.format(i+1, x['human'])
            assistant = target_format.format(x['assistant'])
            utterances += ([human, assistant])

        # utterances:
        # ['[Round 1]\n\n问：如何保障工作中遵循正确的安全准则？\n\n答：',
        #  '为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。',
        #  '[Round 2]\n\n问：如何应对突发紧急情况？\n\n答：',
        #  '突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。',
        #  '[Round 1]\n\n问：如何保障工作中遵循正确的安全准则？\n\n答：',
        #  '为了保障工作中遵循正确的安全准则，以下是一些建议：\n\n1.了解相关安全规定和标准：了解相关的安全规定和标准，并遵守它们。这可以包括公司和政府的安全标准，以及行业标准和最佳实践。\n\n2.培训和教育：确保您和您的同事接受了必要的培训和教育，以了解正确的安全准则和行为。\n\n3.使用正确的工具和设备：确保您使用正确的工具和设备，并且它们得到了正确的维护和保养。\n\n4.个人防护装备：确保您和您的同事穿戴正确的个人防护装备，如安全鞋、透明眼镜或面罩、手套等。\n\n5.创建一个安全文化：鼓励个人对安全的责任感和意识，并创建一个安全文化，使人们始终关注他们自己和他人的安全。\n\n6.持续监测和改进：持续监测和改进安全准则和程序，以确保它们保持最新，并适应变化的情况。\n\n这些措施可以帮助您和您的同事在工作中保障正确的安全准则，确保您的工作场所是一个安全的环境。',
        #  '[Round 2]\n\n问：如何应对突发紧急情况？\n\n答：',
        #  '突发紧急情况可能会在任何时间发生，为了应对这些情况，以下是一些建议：\n\n1.保持冷静：保持冷静和清醒，以便能够更好地应对情况。\n\n2.评估情况：评估紧急情况的性质和范围，以便了解应该采取哪些行动。\n\n3.触发应急计划：如果您的机构或公司有应急计划，立即触发它，并按照计划的步骤行动。\n\n4.联系相关人员：通知必要的人员，例如紧急服务、管理层或其他相关人员，并与他们保持联系。\n\n5.遵循安全准则：在处理紧急情况时，请遵循正确的安全准则，确保自己和其他人的安全。\n\n6.提供及时信息：尽快向相关人员提供准确的信息，以便他们能够做出正确的决策。\n\n7.合作应对：与其他人合作应对紧急情况并相互支持。\n\n总之，应对突发紧急情况需要保持冷静和清醒，评估情况并采取适当的行动。同时，需要遵循正确的安全准则，并与其他人合作应对紧急情况。']
        utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

        # utterances_ids = [[1,2,3],[3,4,5],[2,3,4],[5,6,7]]
        # 每条数据格式为: [Round 1]\n\n问：{input1}\n\n答：{target1}</s>[Round 2]\n\n问：{input2}\n\n答：{target2}</s>...
        input_ids = []
        target_mask = []  # 用于对input进行mask，只计算target部分的loss
        for i, utterances_id in enumerate(utterances_ids):
            # input_ids加入所有内容id
            input_ids += utterances_id

            # input部分 target_mask填补全0
            if i % 2 == 0:
                target_mask += [0] * (len(utterances_id))
            # target部分
            else:
                input_ids += [self.eos_token_id]
                target_mask += [1] * (len(utterances_id) + 1)
        assert len(input_ids) == len(target_mask)

        # 对长度进行截断
        # input_ids:      [1, 2, 3, 3, 4, 5, 'stop', 2, 3, 4, 5, 6, 7, 'stop']
        # target_mask:    [0, 0, 0, 1, 1, 1, 1,      0, 0, 0, 1, 1, 1, 1]
        # attention_mask: [1, 1, 1, 1, 1, 1, 1,      1, 1, 1, 1, 1, 1, 1]
        input_ids = input_ids[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs

