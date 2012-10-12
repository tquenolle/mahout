package org.apache.mahout.utils.vectors.lucene;

import java.io.IOException;
import java.util.*;
import java.io.StringReader;
import java.io.File;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.common.SolrInputField;
import org.apache.solr.common.params.SolrParams;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;
import org.apache.solr.update.AddUpdateCommand;
import org.apache.solr.update.processor.UpdateRequestProcessor;
import org.apache.solr.update.processor.UpdateRequestProcessorFactory;
import org.apache.solr.common.util.NamedList;
import org.apache.solr.analysis.*;
import org.apache.lucene.analysis.fr.ElisionFilter;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.Version.*;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.charfilter.*;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;

import org.apache.solr.update.processor.UpdateRequestProcessor;
import org.apache.solr.update.processor.UpdateRequestProcessorFactory;

import org.apache.mahout.classifier.naivebayes.AbstractNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.vectorizer.encoders.TextValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.utils.vectors.*;

public class CategorizeDocumentFactory extends UpdateRequestProcessorFactory implements Configurable {

    SolrParams params;
    Configuration conf;
    AbstractNaiveBayesClassifier classifier;
    NaiveBayesModel model;
    HashMap<String, Integer> dictMap;
    Map<Integer, String> labelMap;
    String current_cat;
    String outputField;
    int total = 0;
    int correctly = 0;
    int incorrectly = 0;

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
        System.out.println("Initializing Categorize Document Factory");
        this.conf = new Configuration();
        params = SolrParams.toSolrParams((NamedList) args);
        outputField = getOption("outputField");
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
        String[] dict = VectorHelper.loadTermDictionary(new File(getOption("dictFile")));
        dictMap = new HashMap<String, Integer>();
        for (int i = 0; i<dict.length; i++) {
            String val = dict[i];
            if (!dictMap.containsKey(val))
                dictMap.put(val, new Integer(i));
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
            SolrInputDocument doc = cmd.getSolrInputDocument();
            // Allow using multiple fields as input to mimic the copyfield behaviour.
            String inputFields = params.get("inputField");
            String content = getClassificationContent(doc, inputFields);
            String[] tokens = getFilteredTokens(content);
            String guess_cat = classify(tokens);
            doc.addField(outputField, guess_cat);
            super.processAdd(cmd);
        }

        private void print_stats(String f) {
           int real = Integer.parseInt(current_cat);
           int found = Integer.parseInt(f);
           total++;
           if (real == found) {
                correctly++;
           } else {
                incorrectly++;
           }
           System.out.println((((double) correctly/ (double) total)*100) + "%");
        }

        private String classify(String[] ts) {
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
                vec.setQuick((int) idx, (double) termFreqs.get(idx));
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
            System.out.println("Classified as: " + labelMap.get(bestIdx));
            print_stats(labelMap.get(bestIdx));
            return labelMap.get(bestIdx);
        }

        private String getClassificationContent(SolrInputDocument doc, String inputFields) {
            /**
             * Return the the content of the "inputFields" for the doc "doc".
             * inputFields is a coma-separated list of field used for classification.
             */

            StringTokenizer spliter = new StringTokenizer(inputFields, ",");
            String result = "";
            current_cat = ((String) doc.getFieldValue("mail_category"));
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
            StringReader reader = new StringReader(content);
            CharStream cs = CharReader.get(reader);
            HTMLStripCharFilter filter = new HTMLStripCharFilter(cs);
            ArrayList<String> tokenList = new ArrayList<String>(512);
            TokenStream ts = new LowerCaseTokenizer(Version.LUCENE_36, filter);

            ts = new StopFilter(true, ts, StopFilter.makeStopSet(this.stop_words, true), true);
            ts = new ElisionFilter(ts, StopFilter.makeStopSet(this.elision_words, true));
            // Other version of tokenizer.
            while (ts.incrementToken()) {
                tokenList.add(ts.getAttribute(TermAttribute.class).toString());
            }
            String[] tokens = tokenList.toArray(new String[tokenList.size()]);
            return tokens;
        }
    }
}
