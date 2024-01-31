with open('./test_without_label.txt', 'r') as f1:
    lines = f.readlines()

f2 = open('./test2.txt', 'w')
f2.write(lines[0])
for line in lines[1:]:
    guid = line.split(',')[0]
    f1.write(guid)
    f1.write(',')
    label = test_dict[guid]
    print(label)
    if label == 0:
        f2.write('positive\n')
    elif label == 1:
        f2.write('neutral\n')
    else:
        f2.write('negative\n')
