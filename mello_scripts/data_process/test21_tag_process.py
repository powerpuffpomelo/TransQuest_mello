raw_prefix = '/opt/tiger/fake_arnold/mlqe-pe-21/data/test21_goldlabels/en-de-test21/goldlabels/'
save_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/data/test21/en-de-test21/'

for split in ['src', 'mt', 'gaps']:
    raw_file = raw_prefix + 'task2_wordlevel_' + split + '.tags'
    save_file = save_prefix + 'test2021.' + split + '_tag'
    with open(raw_file, 'r', encoding='utf-8') as fr, open(save_file, 'w', encoding='utf-8') as fs:
        save_list = []
        id = 0
        for line in fr.readlines():
            line = line.strip('\n').split('\t')
            if int(line[-4]) > id:
                id = int(line[-4])
                fs.write(' '.join(save_list) + '\n')
                save_list = []
            save_list.append(line[-1])
        fs.write(' '.join(save_list) + '\n')
        

# python3 /opt/tiger/fake_arnold/TransQuest_mello/mello_scripts/data_process/test21_tag_process.py