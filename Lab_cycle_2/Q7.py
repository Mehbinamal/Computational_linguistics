def minimum_edit_distance(source, target, insert_cost=1, delete_cost=1, substitute_cost=1):
    m = len(source)
    n = len(target)
    #Distance matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i * delete_cost
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
    # Fill the distance matrix
    for i in range(1,m+1):
        for j in range(1,n+1):
            if source[i - 1] == target[j-1]:
                substitution_cost = 0
            else:
                substitution_cost = substitute_cost
            
            dp[i][j] = min(
                dp[i - 1][j] + delete_cost,  # Deletion
                dp[i][j - 1] + insert_cost,  # Insertion
                dp[i - 1][j - 1] + substitution_cost  # Substitution
            )

    distance = dp[m][n]

    # Backtrack to find the operations
    operations = []
    i, j = m, n
    while i>0 or j>0:
        if i>0 and j>0 and source[i-1] == target[j-1] and dp[i][j] == dp[i-1][j-1]:
            operations.append(('match', source[i-1], target[j-1]))
            i -= 1
            j -= 1
        elif i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + substitute_cost:
            operations.append(('substitute', source[i-1], target[j-1]))
            i -= 1
            j -= 1
        elif i>0 and dp[i][j] == dp[i-1][j] + delete_cost:
            operations.append(('delete', source[i-1]))
            i -= 1
        else:
            operations.append(('insert', target[j-1]))
            j -= 1
            

    operations.reverse()
    return distance, operations


s1 = "kitten"
s2 = "sitting"
dist, ops = minimum_edit_distance(s1, s2)
print(f"Edit distance between '{s1}' and '{s2}': {dist}")
print("Operations:")
for op in ops:
    print(op)