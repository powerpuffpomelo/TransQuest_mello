# 把mt+gap tag分离

lang_pair = 'en-de'
tag_path_prefix = "/opt/tiger/fake_arnold/wmt-qe-2019-data/test_" + lang_pair + "/"
tag_path = tag_path_prefix + "test.tags"
mt_tag_path = tag_path_prefix + "test.mt_tag"
gap_tag_path = tag_path_prefix + "test.gap_tag"

def read_data(path):
	with open(path, 'r', encoding='utf-8') as f:
		data = f.read()
	return data.split('\n')

tag = read_data(tag_path)
with open(mt_tag_path, 'w', encoding='utf-8') as fmt, open(gap_tag_path, 'w', encoding='utf-8') as fgap:
	for line in tag:
		line = line.split(' ')
		str = ''
		for i, token in enumerate(line):
			if i % 2 == 1:  # 仅保留mt
				str = str + token + ' '
		fmt.write(str.strip()+'\n')
		str = ''
		for i, token in enumerate(line):
			if i % 2 == 0:  # 仅保留gap
				str = str + token + ' '
		fgap.write(str.strip()+'\n')


# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/tool/tag_wo_gap.py