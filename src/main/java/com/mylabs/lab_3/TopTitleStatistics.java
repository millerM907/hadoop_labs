package com.mylabs.lab_3;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Laboratory work 3
 *
 * <p>This class solves some problems:
 * <br>1) counting the number of mentions of each word from the titles-a.txt file (except words from stopwords.txt);
 * <br>2) saving top N words in output file;
 * <br>3) calculating statistical indicators:
 * <br>- arithmetic mean of the number of top N words;
 * <br>- max;
 * <br>- min;
 * <br>- sum;
 * <br>- velocity.
 * <br>The MapReduce algorithm is used to solve this problem.
 *
 * <p>Input data:
 * <br>- titles-b.txt - contains the titles of articles from Wikipedia;
 * <br>- stopwords.txt - contains stop words;
 * <br>- delimiters.txt - contains punctuation marks,
 * <br>- N - is the number of top words stored in the output file (entered from the keyboard).
 *
 * <p>Output data:
 * <br>- file - text file contains result.
 *
 * <p>Example output data for N = 10:
 * <br>Mean	337
 * <br>Sum	1685
 * <br>Min	255
 * <br>Max	461
 * <br>Var	7908
 * */
public class TopTitleStatistics extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TopTitleStatistics(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Configuration conf = this.getConf();
        FileSystem fs = FileSystem.get(conf);
        Path tmpPath = new Path("/lab1/3/tmp");
        fs.delete(tmpPath, true);

        Job jobA = Job.getInstance(conf, "Title Count");
        jobA.setOutputKeyClass(Text.class);
        jobA.setOutputValueClass(IntWritable.class);

        jobA.setMapperClass(TitleCountMap.class);
        jobA.setReducerClass(TitleCountReduce.class);

        FileInputFormat.setInputPaths(jobA, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobA, tmpPath);

        jobA.setJarByClass(TopTitleStatistics.class);
        jobA.waitForCompletion(true);

        Job jobB = Job.getInstance(conf, "Top Titles Statistics");
        jobB.setOutputKeyClass(Text.class);
        jobB.setOutputValueClass(IntWritable.class);

        jobB.setMapOutputKeyClass(NullWritable.class);
        jobB.setMapOutputValueClass(TextArrayWritable.class);

        jobB.setMapperClass(TopTitlesStatMap.class);
        jobB.setReducerClass(TopTitlesStatReduce.class);
        jobB.setNumReduceTasks(1);

        FileInputFormat.setInputPaths(jobB, tmpPath);
        FileOutputFormat.setOutputPath(jobB, new Path(args[1]));

        jobB.setInputFormatClass(KeyValueTextInputFormat.class);
        jobB.setOutputFormatClass(TextOutputFormat.class);

        jobB.setJarByClass(TopTitleStatistics.class);
        return jobB.waitForCompletion(true) ? 0 : 1;
    }

    public static String readHDFSFile(String path, Configuration conf) throws IOException {
        Path pt = new Path(path);
        FileSystem fs = FileSystem.get(pt.toUri(), conf);
        FSDataInputStream file = fs.open(pt);
        BufferedReader buffIn = new BufferedReader(new InputStreamReader(file));

        StringBuilder everything = new StringBuilder();
        String line;
        while ((line = buffIn.readLine()) != null) {
            everything.append(line);
            everything.append("\n");
        }
        return everything.toString();
    }

    public static class TextArrayWritable extends ArrayWritable {
        public TextArrayWritable() {
            super(Text.class);
        }

        public TextArrayWritable(String[] strings) {
            super(Text.class);
            Text[] texts = new Text[strings.length];
            for (int i = 0; i < strings.length; i++) {
                texts[i] = new Text(strings[i]);
            }
            set(texts);
        }
    }

    public static class TitleCountMap extends Mapper<Object, Text, Text, IntWritable> {
        List<String> stopWords;
        String delimiters;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();

            String stopWordsPath = conf.get("stopwords");
            String delimitersPath = conf.get("delimiters");

            this.stopWords = Arrays.asList(readHDFSFile(stopWordsPath, conf).split("\n"));
            this.delimiters = readHDFSFile(delimitersPath, conf);
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            List<String> words = getWords(value);
            List<String> lowerCaseWords = words
                    .stream()
                    .map(String::toLowerCase)
                    .collect(Collectors.toList());
            List<String> filteredWords = deleteStopWordsFromList(lowerCaseWords);
            for (String word : filteredWords) {
                context.write(new Text(word), new IntWritable(1));
            }
        }

        private List<String> getWords(Text value) {
            String backslash = "\\";
            String title = value.toString();
            Set<String> setOfEscapeDelim = new HashSet<>(
                    Arrays.asList("?", "[", "]", "{", "}", "*", ".", "(", ")")
            );
            for (int i = 0; i < delimiters.length(); i++) {
                String delimiter = String.valueOf(delimiters.charAt(i));
                if (title.contains(delimiter)) {
                    if (setOfEscapeDelim.contains(delimiter)) {
                        delimiter = backslash + delimiter;
                    }
                    title = title.replaceAll(delimiter, " ");
                }
            }
            title = title.replaceAll("\\s+", " ").trim();
            String[] words = title.split(" ");
            return Arrays.asList(words);
        }

        private List<String> deleteStopWordsFromList(List<String> words) {
            words.removeIf(word -> stopWords.contains(word));
            return words;
        }
    }

    public static class TitleCountReduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class TopTitlesStatMap extends Mapper<Text, Text, NullWritable, TextArrayWritable> {
        Integer N;
        SortedSet<Pair<Integer, String>> sortedSet = new TreeSet<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            this.N = conf.getInt("N", 10);
        }

        @Override
        public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
            sortedSet.add(new Pair<>(Integer.parseInt(value.toString()), key.toString()));
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            sortedSet.removeIf(elem -> sortedSet.size() > N);
            String[] topWords = sortedSet
                    .stream()
                    .map(elem -> elem.first.toString() + " " + elem.second)
                    .toArray(String[]::new);
            TextArrayWritable textArrayWritable = new TextArrayWritable(topWords);
            context.write(NullWritable.get(), textArrayWritable);
        }
    }

    public static class TopTitlesStatReduce extends Reducer<NullWritable, TextArrayWritable, Text, IntWritable> {
        Integer N;
        SortedSet<Pair<Integer, String>> sortedSet = new TreeSet<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            this.N = conf.getInt("N", 10);
        }

        @Override
        public void reduce(NullWritable key, Iterable<TextArrayWritable> values, Context context) throws IOException, InterruptedException {
            Integer sum, mean, max, min, var;

            addTopNWordsToSet(values);

            sum = getSumTopNWords();
            mean = getMeanTopNWords(sum);
            min = getMinWordsFromTopN();
            max = getMaxWordsFromTopN();
            var = getVarianceTopNWords(mean);

            context.write(new Text("Mean"), new IntWritable(mean));
            context.write(new Text("Sum"), new IntWritable(sum));
            context.write(new Text("Min"), new IntWritable(min));
            context.write(new Text("Max"), new IntWritable(max));
            context.write(new Text("Var"), new IntWritable(var));
        }

        private void addTopNWordsToSet(Iterable<TextArrayWritable> values) {
            for (TextArrayWritable textValues : values) {
                for (Writable writable : textValues.get()) {
                    String[] strArr = writable.toString().split(" ");
                    sortedSet.add(new Pair<>(Integer.parseInt(strArr[0]), strArr[1]));
                }
            }
            sortedSet.removeIf(elem -> sortedSet.size() > N);
        }

        private int getSumTopNWords() {
            int sum = 0;
            for (Pair<Integer, String> pair : sortedSet) {
                sum += pair.first;
            }
            return sum;
        }

        private int getMeanTopNWords(Integer sumTopNWords) {
            return sumTopNWords / N;
        }

        private int getMinWordsFromTopN() {
            return sortedSet.first().first;
        }

        private int getMaxWordsFromTopN() {
            return sortedSet.last().first;
        }

        private int getVarianceTopNWords(Integer mean) {
            List<Double> deviationSquares = new ArrayList<>();
            for (Pair<Integer, String> pair : sortedSet) {
                double deviationN = Math.pow(pair.first - mean, 2);
                deviationSquares.add(deviationN);
            }
            return (int) Math.round(deviationSquares.stream().reduce(Double::sum).get() / N);
        }
    }
}

class Pair<A extends Comparable<? super A>,
        B extends Comparable<? super B>>
        implements Comparable<Pair<A, B>> {

    public final A first;
    public final B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public static <A extends Comparable<? super A>,
            B extends Comparable<? super B>>
    Pair<A, B> of(A first, B second) {
        return new Pair<A, B>(first, second);
    }

    @Override
    public int compareTo(Pair<A, B> o) {
        int cmp = o == null ? 1 : (this.first).compareTo(o.first);
        return cmp == 0 ? (this.second).compareTo(o.second) : cmp;
    }

    @Override
    public int hashCode() {
        return 31 * hashcode(first) + hashcode(second);
    }

    private static int hashcode(Object o) {
        return o == null ? 0 : o.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Pair))
            return false;
        if (this == obj)
            return true;
        return equal(first, ((Pair<?, ?>) obj).first)
                && equal(second, ((Pair<?, ?>) obj).second);
    }

    private boolean equal(Object o1, Object o2) {
        return o1 == o2 || (o1 != null && o1.equals(o2));
    }

    @Override
    public String toString() {
        return "(" + first + ", " + second + ')';
    }
}
