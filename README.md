# masterarbeit-english-intonation
The goal of this research is to explicitly model and test for differences (or similarities) between Majority speaker English and Monolingual speaker English. 

# Requirements
pymer4 
https://eshinjolly.com/pymer4/features.html) -  is an interface to R and requires R to be installed on your system, along with specific R packages such as lme4. The main model used for the generalized binomial mixed model is Lmer for multi-level models estimated using glmer() in R.

            +-------------------+
            |      Python        |   <--- This is where you're working
            +-------------------+
                    |       ^
                    |       |
             Uses  |   Wraps Python around R
                    |       |
        +-----------v-------+-----------+
        |        pymer4 (Python)        |  <--- High-level Python interface for mixed models
        +-------------------------------+
                        |
                        v
           +------------v------------+
           |         rpy2            |   <--- The bridge connecting Python to R
           +-------------------------+
                        |
                        v
             +----------v-----------+
             |          R           |   <--- R environment
             +----------------------+
                        |
                        v
             +----------v-----------+
             |         lme4         |   <--- R package for fitting models
             +----------------------+
