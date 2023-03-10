# hadoop_labs

This project has implemented 3 laboratory works on Apache Hadoop using the MapReduce algorithm.
To solve the laboratory work were used:
- Apache Hadoop 2.9.1
- Apache Hadoop YARN
- Maven
- Java 8

### Laboratory work 1

#### Problem:
It is necessary to count the number of mentions of each word from 
the titles-a.txt file (except words from stopwords.txt).

#### Input data:
- titles-a.txt - contains the titles of articles from Wikipedia;
- stopwords.txt - contains stop words;
- delimiters.txt - contains punctuation marks.

#### Example output data:
world 123
<br>home 34
<br>dream 390

### Laboratory work 2

#### Problem:
It is necessary to count the number of mentions of each word from 
the titles-b.txt file (except words from stopwords.txt),
and then save top N words and their number to txt file.

#### Input data:
 - titles-b.txt - contains the titles of articles from Wikipedia;
 - stopwords.txt - contains stop words;
 - delimiters.txt - contains punctuation marks,
 - N - is the number of top words stored in the output file (entered from the keyboard).
 
#### Output data:
 - file - text file contains result.
 
#### Example output data for N = 4:
 world 38
 <br>home 80
 <br>dream 120
 <br>fly 122

### Laboratory work 3

#### Problem:

It is necessary to solve the following tasks:
- count the number of mentions of each word from the file titles-с.txt
(except for the words from the file stopwords.txt);
- calculate the top N words by the number of mentions;
- save the intermediate result to output txt file;
- calculate statistical indicators: mean, sum, min, max, and velocity.

#### Input data:
- titles-c.txt - contains the titles of articles from Wikipedia;
- stopwords.txt - contains stop words;
- delimiters.txt - contains punctuation marks,
- N - is the number of top words stored in the output file (entered from the keyboard).

#### Output data:
- file - text file contains result.

#### Example output data for N = 10:
Mean	337
<br>Sum	1685
<br>Min	255
<br>Max	461
<br>Var	7908