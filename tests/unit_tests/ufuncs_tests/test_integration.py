import numpy as np
from caspwn.ufuncs.integration import auto_integration


def test_function1():
    function = lambda x: np.exp(-x)
    
    result = auto_integration(function, Ninit=1, rtol=1.0e-15)
    np.testing.assert_allclose(result, 1., rtol=1.0e-15)
        
    result = auto_integration(function, Ninit=10, rtol=1.0e-10)
    np.testing.assert_allclose(result, 1., rtol=1.0e-10)

    result = auto_integration(function, Ninit=5, rtol=1.0e-08)
    np.testing.assert_allclose(result, 1., rtol=1.0e-08)


def test_function2():
    function = lambda x: 2/np.sqrt(np.pi)*np.exp(-x**2)
    
    result = auto_integration(function, Ninit=1, rtol=1.0e-15)
    np.testing.assert_allclose(result, 1., rtol=1.0e-15)
        
    result = auto_integration(function, Ninit=10, rtol=1.0e-10)
    np.testing.assert_allclose(result, 1., rtol=1.0e-10)

    result = auto_integration(function, Ninit=5, rtol=1.0e-08)
    np.testing.assert_allclose(result, 1., rtol=1.0e-08)


def test_function3():
    function = lambda x: 1/(1+x)**2
    
    result = auto_integration(function, Ninit=1, rtol=1.0e-15)
    np.testing.assert_allclose(result, 1., rtol=1.0e-15)
        
    result = auto_integration(function, Ninit=10, rtol=1.0e-10)
    np.testing.assert_allclose(result, 1., rtol=1.0e-10)

    result = auto_integration(function, Ninit=5, rtol=1.0e-08)
    np.testing.assert_allclose(result, 1., rtol=1.0e-08)


def test_function4():
    function = lambda x: 7/(1+x)**8
    
    result = auto_integration(function, Ninit=1, rtol=1.0e-15)
    np.testing.assert_allclose(result, 1., rtol=1.0e-15)
        
    result = auto_integration(function, Ninit=10, rtol=1.0e-10)
    np.testing.assert_allclose(result, 1., rtol=1.0e-10)

    result = auto_integration(function, Ninit=5, rtol=1.0e-08)
    np.testing.assert_allclose(result, 1., rtol=1.0e-08)


if __name__ == "__main__":
    test_function1()
    test_function2()
    test_function3()
    test_function4()
