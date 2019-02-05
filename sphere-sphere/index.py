import numpy as np

def itt(i):
    """
    transforms index to tupel (row_index, column_index)
    
    Parameters
    ----------
    i: integer
        non-negative, index

    Returns
    -------
    (row_index, column_index): tuple

    """
    a = int((np.sqrt(9+8*i)-1)/2)
    last = a*(a+1)//2-1
    if i-last == 0:
        return (a, a-1)
    else:
        return (a+1, i-last-1)


def itt_scalar(n):
    """
    transforms index to tupel (row_index, column_index)
    
    Parameters
    ----------
    n: integer
        non-negative, index

    Returns
    -------
    (row_index, column_index): tuple

    """
    row_index = int((np.sqrt(1+8*n)-1)/2)
    column_index = n - row_index*(row_index+1)//2
    return (row_index, column_index)

def itt_nosquare(index, Nrow, Ncol):
    return index//Ncol, index%Ncol

if __name__ == "__main__":
    print(itt_scalar(5))
