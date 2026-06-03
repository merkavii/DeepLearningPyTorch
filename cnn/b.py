def find_largest(we):
    indx = 0
    for i in range(len(we)):
        if we[i] > we[indx]:
            indx = i
    return we[indx]
print(find_largest([1,2,3,4,7,8,1,9,18]))
