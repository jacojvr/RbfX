# RbfX
Radial basis function implementation [ Python class ] and tests (benchmark / test functions, porper dimensional scaling, cross-validation and Bayesian error measures / inferance)




*******************************************************************************
    A SIMPLE RADIAL BASIS FUNCTION INTERPOLATOR / REGRESSOR:
    
*******************************************************************************
    
    This object takes a multi dimensional input [ x_i ]
                      a one dimensional outpt   [ f_i ] = f(x_i)
                      
    to approximate the function  
                         
         f(x) ~  S(x) =  SUM   [  beta  x phi (x)  + mu  ]
                            k         k      k
                            
    with [ beta ] the kernel weights and [ phi ] the kernel functions.
                            
    The object will regress if fewer RBF kernels [ x_k ] than samples [ x_i ]
    are specified.
    
    The RBF approximated gradient :  d        
                                    --- S(x)
                                     dx
    and estimated uncertainty / variance : ERROR ( S(x) ) are also avalable.
                           
                           
    
*******************************************************************************
