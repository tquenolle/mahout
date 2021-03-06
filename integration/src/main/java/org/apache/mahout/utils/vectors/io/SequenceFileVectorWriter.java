/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils.vectors.io;
import java.util.*;
import java.io.IOException;

import com.google.common.io.Closeables;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.lucene.LuceneIterable;


/**
 * Writes out Vectors to a SequenceFile.
 *
 * Closes the writer when done
 */
public class SequenceFileVectorWriter implements VectorWriter {
  private final SequenceFile.Writer writer;
  private long recNum = 0;
  private static final Set<Integer> EXCLUDE_JOBBOARD_LIST = new HashSet<Integer>(Arrays.asList(new Integer[] {1,3,4,5,6,7,14,15,16,17,23,26,32,33,34,35,36,37,112,125,126,385,745,747,821,1542,1977,2113,2713}));
  public SequenceFileVectorWriter(SequenceFile.Writer writer) {
    this.writer = writer;
  }
  
  @Override
  public long write(Iterable<Vector> iterable, long maxDocs) throws IOException {
    for (Vector point : iterable) {
      if (recNum >= maxDocs) {
        break;
      }
      if (point != null) {
        writer.append(new LongWritable(recNum++), new VectorWritable(point));
      }
      
    }
    return recNum;
  }

  public long writeAndLabel(LuceneIterable iterable, long maxDocs) throws IOException {
    for (Vector point : iterable) {
      if (recNum >= maxDocs) {
        break;
      }
      if (point != null) {
        NamedVector tmp_point = (NamedVector) point;
        String id = tmp_point.getId();
//        if (EXCLUDE_JOBBOARD_LIST.contains(tmp_point.getJobboardId())) {
//            System.out.println("Excluded");
//            continue;
//        }
        if (id != null) {
          writer.append(new Text("/" + tmp_point.getName() + "/" + id), new VectorWritable(point));
        } else {
          writer.append(new Text("/" + tmp_point.getName() + "/" + recNum), new VectorWritable(point));
        }
        recNum++;
      }

    }
    return recNum;
  }

  @Override
  public void write(Vector vector) throws IOException {
    writer.append(new LongWritable(recNum++), new VectorWritable(vector));

  }

  @Override
  public long write(Iterable<Vector> iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }
  
  @Override
  public void close() throws IOException {
    Closeables.closeQuietly(writer);
  }
  
  public SequenceFile.Writer getWriter() {
    return writer;
  }
}
