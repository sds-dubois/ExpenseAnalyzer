def group_indices(data,group_names):
    labels_inv = data['labels_inv']
    group_idx = [labels_inv[label] for label in group_names]
    return(group_idx)

def month_to_string(m):
    month = str(m)
    if(m < 10):
        month = '0' + month
    return(month)