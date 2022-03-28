def intersection(list1,list2):
    return list(set(list1)&set(list2))

def union(list1,list2):
    return list(set(list1)|set(list2))

def union_v2(list1,list2):
    if list1[-2] == list2[0] and list1[-1] == list2[1]:
        return list1+list2[2:]
    elif list2[-2] == list1[0] and list2[-1] == list1[1]:
        return list2+list1[2:] 
    else:
        return "F"

def union_v3(list1,list2,intersec):
    inter1 = retrieve_index(intersec,list1)
    inter2 = retrieve_index(intersec,list2)
    c = compare(inter1,inter2)
    if c[0] != c[1]:
        list2 = list2[::-1]
    success = False
    while not success:
        u = union_v2(list1,list2)
        if u == "F":
            list2 = rotate(list2)
        else:
            success = True
            return u

def retrieve_index(index_list,ring):
    return list(map(lambda x: ring.index(x), index_list))

def compare(index_list1,index_list2):
    return list(map(lambda a,b:a-b, index_list1,index_list2))

def rotate(ring):
    return ring[1:] + [ring[0]]
