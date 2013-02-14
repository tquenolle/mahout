package org.apache.mahout.utils.vectors.lucene;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.io.StringReader;

import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.fr.ElisionFilter;
import org.apache.lucene.analysis.charfilter.*;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.search.DefaultSimilarity;
import org.apache.lucene.search.Similarity;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.Version.*;

import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.vectorizer.encoders.TextValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.TFIDF;
import org.apache.mahout.vectorizer.Weight;
import org.apache.mahout.utils.vectors.*;

import org.apache.solr.analysis.*;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.common.SolrInputField;
import org.apache.solr.common.params.SolrParams;
import org.apache.solr.common.util.NamedList;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;
import org.apache.solr.update.AddUpdateCommand;
import org.apache.solr.update.processor.UpdateRequestProcessor;
import org.apache.solr.update.processor.UpdateRequestProcessorFactory;
import org.apache.solr.update.processor.UpdateRequestProcessor;
import org.apache.solr.update.processor.UpdateRequestProcessorFactory;


public class CategorizeDocumentFactory extends UpdateRequestProcessorFactory implements Configurable {
    /**
     * A class to handle categorization of new Solr documents.
     */

    SolrParams params;
    Configuration conf;
    AbstractNaiveBayesClassifier classifier;
    NaiveBayesModel model;
    HashMap<String, Integer> dictMap;
    HashMap<Integer, Integer> docFreqMap;
    Map<Integer, String> labelMap;
    String current_cat;
    String outputField;
    String modelVersionField;
    String modelVersion;
    String defaultCategory;
    String scoreField;
    boolean debug = false;
    Weight weight = new TFIDF();
    int total = 0;
    int correctly = 0;
    int numDocs;
    private static final Pattern TAB_PATTERN = Pattern.compile("\t");
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public Configuration getConf() {
        return this.conf;
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    private String getOption(String opt) {
        return params.get(opt);
    }

    public void init ( NamedList args ) {
        /**
         * Initialize the update handler.
         */

        System.out.println("Initializing Categorize Document Factory");
        setConf(new Configuration());
        params = SolrParams.toSolrParams((NamedList) args);
        outputField = getOption("outputField");
        modelVersionField = getOption("modelVersionField");
        modelVersion = getOption("modelVersion");
        System.out.println("Using model version: " + modelVersion);
        debug = (getOption("debug").startsWith("true"));
        String defaultCategory = getOption("defaultCategory");
        scoreField = getOption("scoreField");
        try {
            model = NaiveBayesModel.materialize(new Path(getOption("model")), getConf());
            classifier = new StandardNaiveBayesClassifier(model);
            buildDict();
            labelMap = BayesUtils.readLabelIndex(getConf(), new Path(getOption("labelIndex")));
        } catch (IOException e1) {
            e1.printStackTrace();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    private void buildDict() throws IOException {
        /**
         * Cf loadTermDictionary.func
         */

        loadTermDictionary(new FileInputStream(new File(getOption("dictFile"))));
    }

    private void loadTermDictionary(InputStream is) throws IOException {
        /**
         * Read and load a dictionary file.
         * Retrieving:
         *   Number of documents used for tfidf computation.
         * Retrieving for each word:
         *   the feature's index used in the vectors.
         *   it's document frequency (the number of docs it appears in).
         * Inspired by VectorHelper.class
         */

        FileLineIterator it = new FileLineIterator(is);

        int numEntries = Integer.parseInt(it.next());
        dictMap = new HashMap<String, Integer>();
        docFreqMap = new HashMap<Integer, Integer>();

        while (it.hasNext()) {
            String line = it.next();
            if (line.startsWith("#")) {
                if (line.startsWith("#numDocs")) {
                    this.numDocs = Integer.parseInt(SPACE.split(line)[1]);
                }
                continue;
            }
            String[] tokens = TAB_PATTERN.split(line);
            // tokens[0] is the word
            // tokens[1] is the doc freq
            // tokens[2] is the feature index
            if (tokens.length < 3) {
                continue;
            }
            int index = Integer.parseInt(tokens[2]);
            int docfreq = Integer.parseInt(tokens[1]);
            // Saving mapping word -> feature index
            if (!dictMap.containsKey(tokens[0]))
                dictMap.put(tokens[0], new Integer(index));
            // Saving mapping feature index -> doc freq
            if (!docFreqMap.containsKey(tokens[0]))
                docFreqMap.put(new Integer(index), new Integer(docfreq));
        }
    }

    @Override
    public UpdateRequestProcessor getInstance(SolrQueryRequest req, SolrQueryResponse rsp, UpdateRequestProcessor next)
    {
        return new CategorizeDocument(next);
    }

    public class CategorizeDocument extends UpdateRequestProcessor {
        String[] stop_words = {"au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en",
            "et", "eux", "il", "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon",
            "ne", "nos", "notre", "|nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se",
            "ses", "son", "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "c", "d",
            "j", "l", "à", "m", "n", "s", "t", "y", "été", "étée", "étées", "étés", "étant", "suis", "es",
            "est", "sommes", "êtes", "sont", "serai", "seras", "sera", "serons", "serez", "seront", "serais",
            "serait", "serions", "seriez", "seraient", "étais", "était", "étions", "étiez", "étaient", "fus",
            "fut", "fûmes", "fûtes", "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses",
            "fût", "fussions", "fussiez", "fussent", "ayant", "eu", "eue", "eues", "eus", "ai", "as", "avons",
            "avez", "ont", "aurai", "auras", "aura", "aurons", "aurez", "auront", "aurais", "aurait",
            "aurions", "auriez", "auraient", "avais", "avait", "avions", "aviez", "avaient", "eut", "eûmes",
            "eûtes", "eurent", "aie", "aies", "ait", "ayons", "ayez", "aient", "eusse", "eusses", "eût",
            "eussions", "eussiez", "eussent", "ceci", "celà ", "cet", "cette", "ici", "ils", "les", "leurs",
            "quel", "quels", "quelle", "quelles", "sans", "soi"};

        String[] elision_words = {"l", "m", "t", "qu", "n", "s", "j", "d"};

        public CategorizeDocument( UpdateRequestProcessor next) {
            super( next );
        }

        @Override
        public void processAdd(AddUpdateCommand cmd) throws IOException {
            /**
             * Handle new updates.
             * Call the classification computation.
             * Add the document in the index.
             */

            SolrInputDocument doc = cmd.getSolrInputDocument();
            // Allow using multiple fields as input to mimic the copyfield behaviour.
            String inputFields = getOption("inputField");
            String scoreField = getOption("scoreField");

            String guess_cat = defaultCategory;
            String guess_score = "";
            String content = getClassificationContent(doc, inputFields);
            String[] tokens = getFilteredTokens(content);
            if (tokens.length > 0) {
                String class_result[] = classify(tokens);
                guess_cat = class_result[0];
                guess_score = class_result[1];
            }
            doc.addField(outputField, guess_cat);
            doc.addField(scoreField, guess_score);
            doc.addField(modelVersionField, modelVersion);
            super.processAdd(cmd);
        }

        private void stats(String f) {
            /**
             * Increment counters for correctness computation.
             */

            String real = current_cat;
            String found = f;
            total++;
            if (real == found) {
                correctly++;
            }
            if (debug) {
                System.out.println("Was a: " + real);
                System.out.println((((double) correctly/ (double) total)*100) + "%");
            }
        }

        private String[] classify(String[] ts) {
            /**
             * Return the guessed category's label.
             * Term Frequency computation.
             * TFIDF weight computation.
             * Classification based on a model.
             * The best score is returned.
             */

            Map<Integer, Integer> termFreqs = new HashMap<Integer, Integer>();
            for (int k = 0; k<ts.length; k++) {
                String val = ts[k];
                Integer index = dictMap.get(val);
                if (index != null) {
                    if (termFreqs.containsKey(index)) {
                        termFreqs.put(index, termFreqs.get(index) + new Integer(1));
                    } else {
                        termFreqs.put(index, new Integer(1));
                    }
                }
            }
            Vector vec = new RandomAccessSparseVector((int) termFreqs.size());
            for (Integer idx: termFreqs.keySet()) {
                double termWeight = weight.calculate((int)termFreqs.get(idx), (int) docFreqMap.get(idx), 0, numDocs);
                vec.setQuick((int) idx, termWeight);
            }
            Vector scores = classifier.classifyFull(vec.normalize());
            int bestIdx = Integer.MIN_VALUE;
            double bestScore = Long.MIN_VALUE;
            for (Iterator<Vector.Element> score = scores.iterator(); score.hasNext();) {
                Vector.Element element = score.next();
                if (element.get() > bestScore) {
                    bestScore = element.get();
                    bestIdx = element.index();
                }
            }
            if (debug)
                System.out.println("Classified as: " + labelMap.get(bestIdx));
            stats(labelMap.get(bestIdx));
            String result[] = {labelMap.get(bestIdx), String.valueOf(bestScore)};
            return result;
        }

        private String getClassificationContent(SolrInputDocument doc, String inputFields) {
            /**
             * Return the the content of the "inputFields" for the doc "doc".
             * inputFields is a coma-separated list of field used for classification.
             */

            StringTokenizer spliter = new StringTokenizer(inputFields, ",");
            String result = "";
            while (spliter.hasMoreTokens()) {
                String field_name = spliter.nextToken();
                result += getFieldValues(doc, field_name);
            }
            return result;
        }

        private String getFieldValues(SolrInputDocument doc, String inputField) {
            /**
             * Return the value of a given inputField.
             * Supports multivalued fields.
             */

            Collection<Object> input_values = doc.getFieldValues(inputField);
            String input = "";
            if (input_values == null) {
                return input;
            }
            for (Iterator<Object> i = input_values.iterator(); i.hasNext(); ) {
                Object value = i.next();
                if (value != null)
                    input +=  " " + ((String) value);
            }
            return input;
        }

        private String[] getFilteredTokens(String content) throws IOException {
            /**
             * Return an array of the tokenized and filtered input string.
             * Uses:
             *   HTMLStripCharFilter
             *   LowerCaseTokenizer
             *   StopFilter
             *   ElisionFilter
             */

            StringReader reader = new StringReader(content);
            CharStream cs = CharReader.get(reader);
            HTMLStripCharFilter filter = new HTMLStripCharFilter(cs);
            ArrayList<String> tokenList = new ArrayList<String>(512);
            TokenStream ts = new LowerCaseTokenizer(Version.LUCENE_36, filter);

            ts = new StopFilter(true, ts, StopFilter.makeStopSet(this.stop_words, true), true);
            ts = new ElisionFilter(ts, StopFilter.makeStopSet(this.elision_words, true));
            while (ts.incrementToken()) {
                tokenList.add(ts.getAttribute(TermAttribute.class).toString());
            }
            String[] tokens = tokenList.toArray(new String[tokenList.size()]);
            return tokens;
        }
    }
}
