**ALU**

    entity ALU is

        Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);

               b : in  STD_LOGIC_VECTOR (3 downto 0);

               sel : in  STD_LOGIC_VECTOR (2 downto 0);

               led_p : out  STD_LOGIC_VECTOR (3 downto 0);

               led_n : out  STD_LOGIC_VECTOR (3 downto 0));

    end ALU;

    architecture Behavioral of ALU is

    begin

    --led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with sel select

    led_n <= (not (a+b) )when "000",

    (not (a-b)) when "001",

    (not (a and b) ) when "010",

    (not (a nand b)) when "011", 

    (not (a xor b)) when "100",

    (not (a xnor b)) when "101",

    (not (a or b)) when "110",

     (not a) when others;

    end Behavioral;

**JK**

    architecture Behavioral of JK_sph is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with  inp select

    led_n(0) <= (not Q) when "00",

    ('0') when "01",

    ('1') when "10",

    ( Q) when "11",

    'Z' when others;

    end Behavioral;

**Verilog**

    module encoder_8_3(o,i );
    output reg[2:0]o;
    input[7:0]i;
    always@(*)
    case(i)
    8'h01: o=3'b000;
    8'h02: o=3'b001;
    8'h04: o=3'b010;
    8'h08: o=3'b011;
    8'h10: o=3'b100;
    8'h20: o=3'b101;
    8'h40: o=3'b110;
    8'h80: o=3'b111;
    default: o=3'bXXX;
     endcase
    endmodule

**Verilog Test bench**

    module test_bench;
        // Inputs
        reg [7:0] i;

        // Outputs
        wire [2:0] o;

        // Instantiate the Unit Under Test (UUT)
        encoder_8_3 uut (
            .o(o), 
            .i(i)
        );

        initial begin
            // Initialize Inputs
            i = 8'h01;

            // Wait 100 ns for global reset to finish
            #1  i = 8'h02;
            #1 i = 8'h04;
            #1 i = 8'h08;
            #1 i = 8'h10;
            #1 i = 8'h20;
            #1 i = 8'h40;
            #1 i = 8'h80;
        end
        initial #18 $finish;     
    endmodule

**MUX**

    entity mux is
        Port ( sel : in  STD_LOGIC_VECTOR (2 downto 0);
               inp : in  STD_LOGIC_VECTOR (7 downto 0);
               DIS : out  STD_LOGIC_VECTOR (1 downto 0);
               led_p :out STD_LOGIC_VECTOR (3 downto 0);
               led_n :out STD_LOGIC_VECTOR (3 downto 0);
               SEG : out  STD_LOGIC_VECTOR (6 downto 0));
    end mux;

    architecture Behavioral of mux is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    DIS <= "10";

    with sel select

    led_n(0) <= (not inp(0)) when "000",

    (not inp (1)) when "001",

    (not inp (2)) when "010",

    (not inp (3)) when "011",

    (not inp(4)) when "100",

    (not inp (5)) when "101",

    (not inp (6)) when "110",

    (not inp (7)) when "111",

    'Z' when others;

    --Display-- 

    with sel select

    SEG <=

    "1000000" when "000",

    "1111001" when "001",

    "0100100" when "010",

    "0110000" when "011",

    "0011001" when "100",

    "0010010" when "101",

    "0000010" when "110",

    "1111000" when "111",

    "0000000" when others;

    end Behavioral;

**Partity**

    begin


    led_p(3 downto 0) <="0001";

    with sel select

    led_n(3 downto 0) <="1111" when "000",

     "0110" when "001",

     "0101"  when "010",

    "1100" when "011", 

    "0011" when "100",

    "1010" when "101",

    "1001" when "110",
    "0000" when "111",

     "1111" when others;

    end Behavioral;

**SR**

    begin

    led_n (3 downto 1) <= "111";


    led_p(3 downto 0) <="0001";

    with  inp select


    led_n(0) <= (not Q) when "00",


    ('0') when "01",


    ('1') when "10",



    'Z' when others;


    end Behavioral;

# Linear regression

    # importing the dataset

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    dataset = pd.read_csv('Salary_Data.csv')

    dataset.head()

    # data preprocessing

    X = dataset.iloc[:, :-1].values #independent variable array

    y = dataset.iloc[:,1].values #dependent variable vector

    # splitting the dataset

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # fitting the regression model

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()

    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # predicting the test set results

    y_pred = regressor.predict(X_test)

    y_pred

    y_test

    # visualizing the results

    #plot for the TRAIN

    plt.scatter(X_train, y_train, color='red') # plotting the observation line

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Training set)") # stating the title of the graph

    plt.xlabel("Years of experience") # adding the name of x-axis

    plt.ylabel("Salaries") # adding the name of y-axis

    plt.show() # specifies end of graph

    #plot for the TEST

    plt.scatter(X_test, y_test, color='red')

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Testing set)")

    plt.xlabel("Years of experience")

    plt.ylabel("Salaries")

    plt.show()

# Feature Extraction

    #Importing Libraries

    import numpy as np

    import pandas as pd

    import seaborn as sb

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    #reading the dataset

    zomato_real=pd.read_csv("zomato.csv")

    zomato_real.head() # prints the first 5 rows of a DataFrame

    zomato_real.info() # Looking at the information about the dataset, datatypes of the

    coresponding columns and missing values

    #Deleting Unnnecessary Columns

    zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column

    "dish_liked", "phone", "url" and saving the new dataset as "zomato"

    zomato_real.head() # looking at the dataset after transformation

    #Removing the Duplicates

    zomato.duplicated().sum()

    zomato.drop_duplicates(inplace=True)

    zomato_real.head() # looking at the dataset after transformation

    #Remove the NaN values from the dataset

    zomato.isnull().sum()

    zomato.dropna(how='any',inplace=True)

    zomato.info() #.info() function is used to get a concise summary of the dataframe

    #Reading Column Names

    zomato.columns

    #Changing the column names

    zomato = zomato.rename(columns={'approx_cost(for two

    people)':'cost','listed_in(type)':'type',

    'listed_in(city)':'city'})

    zomato.columns

    #Some Transformations

    zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

    zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function

    to replace ',' from cost

    zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

    zomato.info() # looking at the dataset information after transformation

    #Encode the input Variables

    def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

    zomato[column] = zomato[column].factorize()[0]

    return zomato

    zomato_en = Encode(zomato.copy())

    zomato_en.head() # looking at the dataset after transformation

    #Get Correlation between different variables

    corr = zomato_en.corr(method='kendall')

    plt.figure(figsize=(15,8))

    sns.heatmap(corr, annot=True)

    zomato_en.columns**ALU**

    entity ALU is

        Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);

               b : in  STD_LOGIC_VECTOR (3 downto 0);

               sel : in  STD_LOGIC_VECTOR (2 downto 0);

               led_p : out  STD_LOGIC_VECTOR (3 downto 0);

               led_n : out  STD_LOGIC_VECTOR (3 downto 0));

    end ALU;

    architecture Behavioral of ALU is

    begin

    --led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with sel select

    led_n <= (not (a+b) )when "000",

    (not (a-b)) when "001",

    (not (a and b) ) when "010",

    (not (a nand b)) when "011", 

    (not (a xor b)) when "100",

    (not (a xnor b)) when "101",

    (not (a or b)) when "110",

     (not a) when others;

    end Behavioral;

**JK**

    architecture Behavioral of JK_sph is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with  inp select

    led_n(0) <= (not Q) when "00",

    ('0') when "01",

    ('1') when "10",

    ( Q) when "11",

    'Z' when others;

    end Behavioral;

**Verilog**

    module encoder_8_3(o,i );
    output reg[2:0]o;
    input[7:0]i;
    always@(*)
    case(i)
    8'h01: o=3'b000;
    8'h02: o=3'b001;
    8'h04: o=3'b010;
    8'h08: o=3'b011;
    8'h10: o=3'b100;
    8'h20: o=3'b101;
    8'h40: o=3'b110;
    8'h80: o=3'b111;
    default: o=3'bXXX;
     endcase
    endmodule

**Verilog Test bench**

    module test_bench;
        // Inputs
        reg [7:0] i;

        // Outputs
        wire [2:0] o;

        // Instantiate the Unit Under Test (UUT)
        encoder_8_3 uut (
            .o(o), 
            .i(i)
        );

        initial begin
            // Initialize Inputs
            i = 8'h01;

            // Wait 100 ns for global reset to finish
            #1  i = 8'h02;
            #1 i = 8'h04;
            #1 i = 8'h08;
            #1 i = 8'h10;
            #1 i = 8'h20;
            #1 i = 8'h40;
            #1 i = 8'h80;
        end
        initial #18 $finish;     
    endmodule

**MUX**

    entity mux is
        Port ( sel : in  STD_LOGIC_VECTOR (2 downto 0);
               inp : in  STD_LOGIC_VECTOR (7 downto 0);
               DIS : out  STD_LOGIC_VECTOR (1 downto 0);
               led_p :out STD_LOGIC_VECTOR (3 downto 0);
               led_n :out STD_LOGIC_VECTOR (3 downto 0);
               SEG : out  STD_LOGIC_VECTOR (6 downto 0));
    end mux;

    architecture Behavioral of mux is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    DIS <= "10";

    with sel select

    led_n(0) <= (not inp(0)) when "000",

    (not inp (1)) when "001",

    (not inp (2)) when "010",

    (not inp (3)) when "011",

    (not inp(4)) when "100",

    (not inp (5)) when "101",

    (not inp (6)) when "110",

    (not inp (7)) when "111",

    'Z' when others;

    --Display-- 

    with sel select

    SEG <=

    "1000000" when "000",

    "1111001" when "001",

    "0100100" when "010",

    "0110000" when "011",

    "0011001" when "100",

    "0010010" when "101",

    "0000010" when "110",

    "1111000" when "111",

    "0000000" when others;

    end Behavioral;

**Partity**

    begin


    led_p(3 downto 0) <="0001";

    with sel select

    led_n(3 downto 0) <="1111" when "000",

     "0110" when "001",

     "0101"  when "010",

    "1100" when "011", 

    "0011" when "100",

    "1010" when "101",

    "1001" when "110",
    "0000" when "111",

     "1111" when others;

    end Behavioral;

**SR**

    begin

    led_n (3 downto 1) <= "111";


    led_p(3 downto 0) <="0001";

    with  inp select


    led_n(0) <= (not Q) when "00",


    ('0') when "01",


    ('1') when "10",



    'Z' when others;


    end Behavioral;

# Linear regression

    # importing the dataset

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    dataset = pd.read_csv('Salary_Data.csv')

    dataset.head()

    # data preprocessing

    X = dataset.iloc[:, :-1].values #independent variable array

    y = dataset.iloc[:,1].values #dependent variable vector

    # splitting the dataset

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # fitting the regression model

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()

    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # predicting the test set results

    y_pred = regressor.predict(X_test)

    y_pred

    y_test

    # visualizing the results

    #plot for the TRAIN

    plt.scatter(X_train, y_train, color='red') # plotting the observation line

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Training set)") # stating the title of the graph

    plt.xlabel("Years of experience") # adding the name of x-axis

    plt.ylabel("Salaries") # adding the name of y-axis

    plt.show() # specifies end of graph

    #plot for the TEST

    plt.scatter(X_test, y_test, color='red')

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Testing set)")

    plt.xlabel("Years of experience")

    plt.ylabel("Salaries")

    plt.show()

# Feature Extraction

    #Importing Libraries

    import numpy as np

    import pandas as pd

    import seaborn as sb

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    #reading the dataset

    zomato_real=pd.read_csv("zomato.csv")

    zomato_real.head() # prints the first 5 rows of a DataFrame

    zomato_real.info() # Looking at the information about the dataset, datatypes of the

    coresponding columns and missing values

    #Deleting Unnnecessary Columns

    zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column

    "dish_liked", "phone", "url" and saving the new dataset as "zomato"

    zomato_real.head() # looking at the dataset after transformation

    #Removing the Duplicates

    zomato.duplicated().sum()

    zomato.drop_duplicates(inplace=True)

    zomato_real.head() # looking at the dataset after transformation

    #Remove the NaN values from the dataset

    zomato.isnull().sum()

    zomato.dropna(how='any',inplace=True)

    zomato.info() #.info() function is used to get a concise summary of the dataframe

    #Reading Column Names

    zomato.columns

    #Changing the column names

    zomato = zomato.rename(columns={'approx_cost(for two

    people)':'cost','listed_in(type)':'type',

    'listed_in(city)':'city'})

    zomato.columns

    #Some Transformations

    zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

    zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function

    to replace ',' from cost

    zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

    zomato.info() # looking at the dataset information after transformation

    #Encode the input Variables

    def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

    zomato[column] = zomato[column].factorize()[0]

    return zomato

    zomato_en = Encode(zomato.copy())

    zomato_en.head() # looking at the dataset after transformation

    #Get Correlation between different variables

    corr = zomato_en.corr(method='kendall')

    plt.figure(figsize=(15,8))
**ALU**

    entity ALU is

        Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);

               b : in  STD_LOGIC_VECTOR (3 downto 0);

               sel : in  STD_LOGIC_VECTOR (2 downto 0);

               led_p : out  STD_LOGIC_VECTOR (3 downto 0);

               led_n : out  STD_LOGIC_VECTOR (3 downto 0));

    end ALU;

    architecture Behavioral of ALU is

    begin

    --led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with sel select

    led_n <= (not (a+b) )when "000",

    (not (a-b)) when "001",

    (not (a and b) ) when "010",

    (not (a nand b)) when "011", 

    (not (a xor b)) when "100",

    (not (a xnor b)) when "101",

    (not (a or b)) when "110",

     (not a) when others;

    end Behavioral;

**JK**

    architecture Behavioral of JK_sph is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with  inp select

    led_n(0) <= (not Q) when "00",

    ('0') when "01",

    ('1') when "10",

    ( Q) when "11",

    'Z' when others;

    end Behavioral;

**Verilog**

    module encoder_8_3(o,i );
    output reg[2:0]o;
    input[7:0]i;
    always@(*)
    case(i)
    8'h01: o=3'b000;
    8'h02: o=3'b001;
    8'h04: o=3'b010;
    8'h08: o=3'b011;
    8'h10: o=3'b100;
    8'h20: o=3'b101;
    8'h40: o=3'b110;
    8'h80: o=3'b111;
    default: o=3'bXXX;
     endcase
    endmodule

**Verilog Test bench**

    module test_bench;
        // Inputs
        reg [7:0] i;

        // Outputs
        wire [2:0] o;

        // Instantiate the Unit Under Test (UUT)
        encoder_8_3 uut (
            .o(o), 
            .i(i)
        );

        initial begin
            // Initialize Inputs
            i = 8'h01;

            // Wait 100 ns for global reset to finish
            #1  i = 8'h02;
            #1 i = 8'h04;
            #1 i = 8'h08;
            #1 i = 8'h10;
            #1 i = 8'h20;
            #1 i = 8'h40;
            #1 i = 8'h80;
        end
        initial #18 $finish;     
    endmodule

**MUX**

    entity mux is
        Port ( sel : in  STD_LOGIC_VECTOR (2 downto 0);
               inp : in  STD_LOGIC_VECTOR (7 downto 0);
               DIS : out  STD_LOGIC_VECTOR (1 downto 0);
               led_p :out STD_LOGIC_VECTOR (3 downto 0);
               led_n :out STD_LOGIC_VECTOR (3 downto 0);
               SEG : out  STD_LOGIC_VECTOR (6 downto 0));
    end mux;

    architecture Behavioral of mux is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    DIS <= "10";

    with sel select

    led_n(0) <= (not inp(0)) when "000",

    (not inp (1)) when "001",

    (not inp (2)) when "010",

    (not inp (3)) when "011",

    (not inp(4)) when "100",

    (not inp (5)) when "101",

    (not inp (6)) when "110",

    (not inp (7)) when "111",

    'Z' when others;

    --Display-- 

    with sel select

    SEG <=

    "1000000" when "000",

    "1111001" when "001",

    "0100100" when "010",

    "0110000" when "011",

    "0011001" when "100",

    "0010010" when "101",

    "0000010" when "110",

    "1111000" when "111",

    "0000000" when others;

    end Behavioral;

**Partity**

    begin


    led_p(3 downto 0) <="0001";

    with sel select

    led_n(3 downto 0) <="1111" when "000",

     "0110" when "001",

     "0101"  when "010",

    "1100" when "011", 

    "0011" when "100",

    "1010" when "101",

    "1001" when "110",
    "0000" when "111",

     "1111" when others;

    end Behavioral;

**SR**

    begin

    led_n (3 downto 1) <= "111";


    led_p(3 downto 0) <="0001";

    with  inp select


    led_n(0) <= (not Q) when "00",


    ('0') when "01",


    ('1') when "10",



    'Z' when others;


    end Behavioral;

# Linear regression

    # importing the dataset

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    dataset = pd.read_csv('Salary_Data.csv')

    dataset.head()

    # data preprocessing

    X = dataset.iloc[:, :-1].values #independent variable array

    y = dataset.iloc[:,1].values #dependent variable vector

    # splitting the dataset

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # fitting the regression model

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()

    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # predicting the test set results

    y_pred = regressor.predict(X_test)

    y_pred

    y_test

    # visualizing the results

    #plot for the TRAIN

    plt.scatter(X_train, y_train, color='red') # plotting the observation line

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Training set)") # stating the title of the graph

    plt.xlabel("Years of experience") # adding the name of x-axis

    plt.ylabel("Salaries") # adding the name of y-axis

    plt.show() # specifies end of graph

    #plot for the TEST

    plt.scatter(X_test, y_test, color='red')

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Testing set)")

    plt.xlabel("Years of experience")

    plt.ylabel("Salaries")

    plt.show()

# Feature Extraction

    #Importing Libraries

    import numpy as np

    import pandas as pd

    import seaborn as sb

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    #reading the dataset

    zomato_real=pd.read_csv("zomato.csv")

    zomato_real.head() # prints the first 5 rows of a DataFrame

    zomato_real.info() # Looking at the information about the dataset, datatypes of the

    coresponding columns and missing values

    #Deleting Unnnecessary Columns

    zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column

    "dish_liked", "phone", "url" and saving the new dataset as "zomato"

    zomato_real.head() # looking at the dataset after transformation

    #Removing the Duplicates

    zomato.duplicated().sum()

    zomato.drop_duplicates(inplace=True)

    zomato_real.head() # looking at the dataset after transformation

    #Remove the NaN values from the dataset

    zomato.isnull().sum()

    zomato.dropna(how='any',inplace=True)

    zomato.info() #.info() function is used to get a concise summary of the dataframe

    #Reading Column Names

    zomato.columns

    #Changing the column names

    zomato = zomato.rename(columns={'approx_cost(for two

    people)':'cost','listed_in(type)':'type',

    'listed_in(city)':'city'})

    zomato.columns

    #Some Transformations

    zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

    zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function

    to replace ',' from cost

    zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

    zomato.info() # looking at the dataset information after transformation

    #Encode the input Variables

    def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

    zomato[column] = zomato[column].factorize()[0]

    return zomato

    zomato_en = Encode(zomato.copy())

    zomato_en.head() # looking at the dataset after transformation

    #Get Correlation between different variables

    corr = zomato_en.corr(method='kendall')

    plt.figure(figsize=(15,8))

    sns.heatmap(corr, annot=True)

    zomato_en.columns**ALU**

    entity ALU is

        Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);

               b : in  STD_LOGIC_VECTOR (3 downto 0);

               sel : in  STD_LOGIC_VECTOR (2 downto 0);

               led_p : out  STD_LOGIC_VECTOR (3 downto 0);

               led_n : out  STD_LOGIC_VECTOR (3 downto 0));

    end ALU;

    architecture Behavioral of ALU is

    begin

    --led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with sel select

    led_n <= (not (a+b) )when "000",

    (not (a-b)) when "001",

    (not (a and b) ) when "010",

    (not (a nand b)) when "011", 

    (not (a xor b)) when "100",

    (not (a xnor b)) when "101",

    (not (a or b)) when "110",

     (not a) when others;

    end Behavioral;

**JK**

    architecture Behavioral of JK_sph is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with  inp select

    led_n(0) <= (not Q) when "00",

    ('0') when "01",

    ('1') when "10",

    ( Q) when "11",

    'Z' when others;

    end Behavioral;

**Verilog**

    module encoder_8_3(o,i );
    output reg[2:0]o;
    input[7:0]i;
    always@(*)
    case(i)
    8'h01: o=3'b000;
    8'h02: o=3'b001;
    8'h04: o=3'b010;
    8'h08: o=3'b011;
    8'h10: o=3'b100;
    8'h20: o=3'b101;
    8'h40: o=3'b110;
    8'h80: o=3'b111;
    default: o=3'bXXX;
     endcase
    endmodule

**Verilog Test bench**

    module test_bench;
        // Inputs
        reg [7:0] i;

        // Outputs
        wire [2:0] o;

        // Instantiate the Unit Under Test (UUT)
        encoder_8_3 uut (
            .o(o), 
            .i(i)
        );

        initial begin
            // Initialize Inputs
            i = 8'h01;

            // Wait 100 ns for global reset to finish
            #1  i = 8'h02;
            #1 i = 8'h04;
            #1 i = 8'h08;
            #1 i = 8'h10;
            #1 i = 8'h20;
            #1 i = 8'h40;
            #1 i = 8'h80;
        end
        initial #18 $finish;     
    endmodule

**MUX**

    entity mux is
        Port ( sel : in  STD_LOGIC_VECTOR (2 downto 0);
               inp : in  STD_LOGIC_VECTOR (7 downto 0);
               DIS : out  STD_LOGIC_VECTOR (1 downto 0);
               led_p :out STD_LOGIC_VECTOR (3 downto 0);
               led_n :out STD_LOGIC_VECTOR (3 downto 0);
               SEG : out  STD_LOGIC_VECTOR (6 downto 0));
    end mux;

    architecture Behavioral of mux is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    DIS <= "10";

    with sel select

    led_n(0) <= (not inp(0)) when "000",

    (not inp (1)) when "001",

    (not inp (2)) when "010",

    (not inp (3)) when "011",

    (not inp(4)) when "100",

    (not inp (5)) when "101",

    (not inp (6)) when "110",

    (not inp (7)) when "111",

    'Z' when others;

    --Display-- 

    with sel select

    SEG <=

    "1000000" when "000",

    "1111001" when "001",

    "0100100" when "010",

    "0110000" when "011",

    "0011001" when "100",

    "0010010" when "101",

    "0000010" when "110",

    "1111000" when "111",

    "0000000" when others;

    end Behavioral;

**Partity**

    begin


    led_p(3 downto 0) <="0001";

    with sel select

    led_n(3 downto 0) <="1111" when "000",

     "0110" when "001",

     "0101"  when "010",

    "1100" when "011", 

    "0011" when "100",

    "1010" when "101",

    "1001" when "110",
    "0000" when "111",

     "1111" when others;

    end Behavioral;

**SR**

    begin

    led_n (3 downto 1) <= "111";


    led_p(3 downto 0) <="0001";

    with  inp select


    led_n(0) <= (not Q) when "00",


    ('0') when "01",


    ('1') when "10",



    'Z' when others;


    end Behavioral;

# Linear regression

    # importing the dataset

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    dataset = pd.read_csv('Salary_Data.csv')

    dataset.head()

    # data preprocessing

    X = dataset.iloc[:, :-1].values #independent variable array

    y = dataset.iloc[:,1].values #dependent variable vector

    # splitting the dataset

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # fitting the regression model

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()

    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # predicting the test set results

    y_pred = regressor.predict(X_test)

    y_pred

    y_test

    # visualizing the results

    #plot for the TRAIN

    plt.scatter(X_train, y_train, color='red') # plotting the observation line

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Training set)") # stating the title of the graph

    plt.xlabel("Years of experience") # adding the name of x-axis

    plt.ylabel("Salaries") # adding the name of y-axis

    plt.show() # specifies end of graph

    #plot for the TEST

    plt.scatter(X_test, y_test, color='red')

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Testing set)")

    plt.xlabel("Years of experience")

    plt.ylabel("Salaries")

    plt.show()

# Feature Extraction

    #Importing Libraries

    import numpy as np

    import pandas as pd

    import seaborn as sb

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    #reading the dataset

    zomato_real=pd.read_csv("zomato.csv")

    zomato_real.head() # prints the first 5 rows of a DataFrame

    zomato_real.info() # Looking at the information about the dataset, datatypes of the

    coresponding columns and missing values

    #Deleting Unnnecessary Columns

    zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column

    "dish_liked", "phone", "url" and saving the new dataset as "zomato"

    zomato_real.head() # looking at the dataset after transformation

    #Removing the Duplicates

    zomato.duplicated().sum()

    zomato.drop_duplicates(inplace=True)

    zomato_real.head() # looking at the dataset after transformation

    #Remove the NaN values from the dataset

    zomato.isnull().sum()

    zomato.dropna(how='any',inplace=True)

    zomato.info() #.info() function is used to get a concise summary of the dataframe

    #Reading Column Names

    zomato.columns

    #Changing the column names

    zomato = zomato.rename(columns={'approx_cost(for two

    people)':'cost','listed_in(type)':'type',

    'listed_in(city)':'city'})

    zomato.columns

    #Some Transformations

    zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

    zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function

    to replace ',' from cost

    zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

    zomato.info() # looking at the dataset information after transformation

    #Encode the input Variables

    def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

    zomato[column] = zomato[column].factorize()[0]

    return zomato

    zomato_en = Encode(zomato.copy())

    zomato_en.head() # looking at the dataset after transformation

    #Get Correlation between different variables

    corr = zomato_en.corr(method='kendall')

    plt.figure(figsize=(15,8))

    sns.heatmap(corr, annot=True)

    zomato_en.columns**ALU**

    entity ALU is

        Port ( a : in  STD_LOGIC_VECTOR (3 downto 0);

               b : in  STD_LOGIC_VECTOR (3 downto 0);

               sel : in  STD_LOGIC_VECTOR (2 downto 0);

               led_p : out  STD_LOGIC_VECTOR (3 downto 0);

               led_n : out  STD_LOGIC_VECTOR (3 downto 0));

    end ALU;

    architecture Behavioral of ALU is

    begin

    --led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with sel select

    led_n <= (not (a+b) )when "000",

    (not (a-b)) when "001",

    (not (a and b) ) when "010",

    (not (a nand b)) when "011", 

    (not (a xor b)) when "100",

    (not (a xnor b)) when "101",

    (not (a or b)) when "110",

     (not a) when others;

    end Behavioral;

**JK**

    architecture Behavioral of JK_sph is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    with  inp select

    led_n(0) <= (not Q) when "00",

    ('0') when "01",

    ('1') when "10",

    ( Q) when "11",

    'Z' when others;

    end Behavioral;

**Verilog**

    module encoder_8_3(o,i );
    output reg[2:0]o;
    input[7:0]i;
    always@(*)
    case(i)
    8'h01: o=3'b000;
    8'h02: o=3'b001;
    8'h04: o=3'b010;
    8'h08: o=3'b011;
    8'h10: o=3'b100;
    8'h20: o=3'b101;
    8'h40: o=3'b110;
    8'h80: o=3'b111;
    default: o=3'bXXX;
     endcase
    endmodule

**Verilog Test bench**

    module test_bench;
        // Inputs
        reg [7:0] i;

        // Outputs
        wire [2:0] o;

        // Instantiate the Unit Under Test (UUT)
        encoder_8_3 uut (
            .o(o), 
            .i(i)
        );

        initial begin
            // Initialize Inputs
            i = 8'h01;

            // Wait 100 ns for global reset to finish
            #1  i = 8'h02;
            #1 i = 8'h04;
            #1 i = 8'h08;
            #1 i = 8'h10;
            #1 i = 8'h20;
            #1 i = 8'h40;
            #1 i = 8'h80;
        end
        initial #18 $finish;     
    endmodule

**MUX**

    entity mux is
        Port ( sel : in  STD_LOGIC_VECTOR (2 downto 0);
               inp : in  STD_LOGIC_VECTOR (7 downto 0);
               DIS : out  STD_LOGIC_VECTOR (1 downto 0);
               led_p :out STD_LOGIC_VECTOR (3 downto 0);
               led_n :out STD_LOGIC_VECTOR (3 downto 0);
               SEG : out  STD_LOGIC_VECTOR (6 downto 0));
    end mux;

    architecture Behavioral of mux is

    begin

    led_n (3 downto 1) <= "111";

    led_p(3 downto 0) <="0001";

    DIS <= "10";

    with sel select

    led_n(0) <= (not inp(0)) when "000",

    (not inp (1)) when "001",

    (not inp (2)) when "010",

    (not inp (3)) when "011",

    (not inp(4)) when "100",

    (not inp (5)) when "101",

    (not inp (6)) when "110",

    (not inp (7)) when "111",

    'Z' when others;

    --Display-- 

    with sel select

    SEG <=

    "1000000" when "000",

    "1111001" when "001",

    "0100100" when "010",

    "0110000" when "011",

    "0011001" when "100",

    "0010010" when "101",

    "0000010" when "110",

    "1111000" when "111",

    "0000000" when others;

    end Behavioral;

**Partity**

    begin


    led_p(3 downto 0) <="0001";

    with sel select

    led_n(3 downto 0) <="1111" when "000",

     "0110" when "001",

     "0101"  when "010",

    "1100" when "011", 

    "0011" when "100",

    "1010" when "101",

    "1001" when "110",
    "0000" when "111",

     "1111" when others;

    end Behavioral;

**SR**

    begin

    led_n (3 downto 1) <= "111";


    led_p(3 downto 0) <="0001";

    with  inp select


    led_n(0) <= (not Q) when "00",


    ('0') when "01",


    ('1') when "10",



    'Z' when others;


    end Behavioral;

# Linear regression

    # importing the dataset

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    dataset = pd.read_csv('Salary_Data.csv')

    dataset.head()

    # data preprocessing

    X = dataset.iloc[:, :-1].values #independent variable array

    y = dataset.iloc[:,1].values #dependent variable vector

    # splitting the dataset

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)

    # fitting the regression model

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()

    regressor.fit(X_train,y_train) #actually produces the linear eqn for the data

    # predicting the test set results

    y_pred = regressor.predict(X_test)

    y_pred

    y_test

    # visualizing the results

    #plot for the TRAIN

    plt.scatter(X_train, y_train, color='red') # plotting the observation line

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Training set)") # stating the title of the graph

    plt.xlabel("Years of experience") # adding the name of x-axis

    plt.ylabel("Salaries") # adding the name of y-axis

    plt.show() # specifies end of graph

    #plot for the TEST

    plt.scatter(X_test, y_test, color='red')

    plt.plot(X_train, regressor.predict(X_train), color='blue') # plotting the regression line

    plt.title("Salary vs Experience (Testing set)")

    plt.xlabel("Years of experience")

    plt.ylabel("Salaries")

    plt.show()

# Feature Extraction

    #Importing Libraries

    import numpy as np

    import pandas as pd

    import seaborn as sb

    import matplotlib.pyplot as plt

    import seaborn as sns

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LinearRegression

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import classification_report

    from sklearn.metrics import confusion_matrix

    #reading the dataset

    zomato_real=pd.read_csv("zomato.csv")

    zomato_real.head() # prints the first 5 rows of a DataFrame

    zomato_real.info() # Looking at the information about the dataset, datatypes of the

    coresponding columns and missing values

    #Deleting Unnnecessary Columns

    zomato=zomato_real.drop(['url','dish_liked','phone'],axis=1) #Dropping the column

    "dish_liked", "phone", "url" and saving the new dataset as "zomato"

    zomato_real.head() # looking at the dataset after transformation

    #Removing the Duplicates

    zomato.duplicated().sum()

    zomato.drop_duplicates(inplace=True)

    zomato_real.head() # looking at the dataset after transformation

    #Remove the NaN values from the dataset

    zomato.isnull().sum()

    zomato.dropna(how='any',inplace=True)

    zomato.info() #.info() function is used to get a concise summary of the dataframe

    #Reading Column Names

    zomato.columns

    #Changing the column names

    zomato = zomato.rename(columns={'approx_cost(for two

    people)':'cost','listed_in(type)':'type',

    'listed_in(city)':'city'})

    zomato.columns

    #Some Transformations

    zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string

    zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function

    to replace ',' from cost

    zomato['cost'] = zomato['cost'].astype(float) # Changing the cost to Float

    zomato.info() # looking at the dataset information after transformation

    #Encode the input Variables

    def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

    zomato[column] = zomato[column].factorize()[0]

    return zomato

    zomato_en = Encode(zomato.copy())

    zomato_en.head() # looking at the dataset after transformation

    #Get Correlation between different variables

    corr = zomato_en.corr(method='kendall')

    plt.figure(figsize=(15,8))

    sns.heatmap(corr, annot=True)

    zomato_en.columns
    sns.heatmap(corr, annot=True)

    zomato_en.columns
