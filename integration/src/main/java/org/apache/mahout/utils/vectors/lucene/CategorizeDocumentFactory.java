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
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.DenseVector;

public class CategorizeDocumentFactory extends UpdateRequestProcessorFactory implements Configurable {

    SolrParams params;
    Configuration conf;
    AbstractNaiveBayesClassifier classifier;
    NaiveBayesModel model;

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
        try {
            model = NaiveBayesModel.materialize(new Path(getOption("model")), getConf());
            classifier = new StandardNaiveBayesClassifier(model);
        } catch (IOException e1) {
            e1.printStackTrace();
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public UpdateRequestProcessor getInstance(SolrQueryRequest req, SolrQueryResponse rsp, UpdateRequestProcessor next)
    {
        return new CategorizeDocument(next);
    }

    public class CategorizeDocument extends UpdateRequestProcessor {

        public CategorizeDocument( UpdateRequestProcessor next) {
            super( next );
            System.out.println("INIT Categorize doc");
        }

        @Override
        public void processAdd(AddUpdateCommand cmd) throws IOException {
            System.out.println("Process ADD !");
            SolrInputDocument doc = cmd.getSolrInputDocument();
            // Allow using multiple fields as input to mimic the copyfield behaviour.
            String inputFields = params.get("inputField");
            String content = getClassificationContent(doc, inputFields);
            String[] tokens = getFilteredTokens(content);
            System.out.println("Ready to classify");
            for (int k = 0; k<tokens.length; k++) {
                System.out.println(tokens[k]);
            }
            classify(tokens);
        }

        private void classify(String[] ts) {
            Vector vec = new DenseVector((int) model.numFeatures());
            System.out.println(model.numFeatures());
            TextValueEncoder encoder = new TextValueEncoder("encoder");
            for (int i = 0; i < ts.length; i++) {
                encoder.addText(ts[i]);
            }
            encoder.flush(1.0, vec);
            System.out.println("VECTOR !");
            System.out.println(vec);
            System.out.println("NUM CAT:" + classifier.numCategories());
            System.out.println(classifier.classifyFull(vec));
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
                System.out.println("Extracting: " + field_name);
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
                System.out.println("Input NULL !");
                return input;
            }
            for (Iterator<Object> i = input_values.iterator(); i.hasNext(); ) {
                Object value = i.next();
                System.out.println((String) value);
                if (value != null)
                    input +=  " " + ((String) value);
            }
            return input;
        }

        private String[] getFilteredTokens(String content) throws IOException {
            StringReader reader = new StringReader(content);
            CharStream cs = CharReader.get(reader);
            HTMLStripCharFilter filter = new HTMLStripCharFilter(cs);
            ArrayList<String> tokenList = new ArrayList<String>(256);
            StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_36);
            TokenStream ts = analyzer.tokenStream(null, filter);
            // Other version of tokenizer.
//            TokenStream ts = new LowerCaseTokenizer(Version.LUCENE_36, filter);
            while (ts.incrementToken()) {
                tokenList.add(ts.getAttribute(TermAttribute.class).toString());
            }
            String[] tokens = tokenList.toArray(new String[tokenList.size()]);
            return tokens;
        }
    }
}


