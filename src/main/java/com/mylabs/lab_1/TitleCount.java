package com.mylabs.lab_1;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Laboratory work 1
 *
 * <p>This class solves the problem of counting the number of mentions
 * <br>of each word from the titles-a.txt file (except words from stopwords.txt).
 * <br>The MapReduce algorithm is used to solve this problem.
 *
 * <p>Input data:
 * <br>- titles-a.txt - contains the titles of articles from Wikipedia;
 * <br>- stopwords.txt - contains stop words;
 * <br>- delimiters.txt - contains punctuation marks.
 *
 * <p>Output data:
 * <br>- file - text file contains result.
 *
 * <p>Example output data:
 * <br>world 123
 * <br>home 34
 * <br>dream 390
 * */
public class TitleCount extends Configured implements Tool {
    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new Configuration(), new TitleCount(), args);
        System.exit(res);
    }

    @Override
    public int run(String[] args) throws Exception {
        Job job = Job.getInstance(this.getConf(), "Title Count");
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setMapperClass(TitleCountMap.class);
        job.setReducerClass(TitleCountReduce.class);

        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.setJarByClass(TitleCount.class);
        return job.waitForCompletion(true) ? 0 : 1;
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

    public static class TitleCountMap extends Mapper<Object, Text, Text, IntWritable> {
        List<String> stopWords;
        String delimiters;
        IntWritable one = new IntWritable(1);

        @Override
        protected void setup(Context context) throws IOException {

            Configuration conf = context.getConfiguration();

            String stopWordsPath = conf.get("stopwords");
            String delimitersPath = conf.get("delimiters");

            this.stopWords = Arrays.asList(readHDFSFile(stopWordsPath, conf).split("\n"));
            this.delimiters = readHDFSFile(delimitersPath, conf);
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            List<String> words = getWords(value);
            List<String> lowerCaseWords = words.stream().map(String::toLowerCase).collect(Collectors.toList());
            List<String> filteredWords = deleteStopWordsFromList(lowerCaseWords);
            for (String word : filteredWords) {
                context.write(new Text(word), one);
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
}