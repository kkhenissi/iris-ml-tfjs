import { Component, OnInit, AfterViewInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { THIS_EXPR } from '@angular/compiler/src/output/output_ast';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit  {

  public tensors: tf.Tensor2D[] = [];
  public model: tf.Sequential;
  public learningRate: number = 0.001;
  public modelCreated: boolean;

  ngOnInit(): void {
    this.tensors[ 'X'] = tf.tensor2d([45, 11, 25, 31, 45, 26, 45, 21, 22, 15, 44, 66], [3, 4]);
    this.tensors[ 'Y'] = tf.tensor2d([[5, 44, 7], [5, 15, 21], [74, 58, 47], [95, 42, 84] ]);
    this.tensors['X'].print();
    this.tensors['Y'].print();

  }

  onMult() {
    this.tensors['Z'] = this.tensors['X'].matMul(this.tensors['Y']);
  }

  onTranspose() {
     this.tensors['Z']=this.tensors['X'].transpose();
  }
  onSigmoide() {
    this.tensors['Z']=tf.sigmoid(this.tensors['X']);
  }
  onRelu() {
    this.tensors['Z']=tf.relu(this.tensors['X']);

  }
  onCreateModel() {
  
    this.model = tf.sequential();

    this.model.add(tf.layers.dense({
      units:10,
      activation: 'sigmoid',
      inputShape: [4]
    }));

    this.model.add(tf.layers.dense({
      units:3,
      activation: 'softmax'
    }));

    this.model.compile({
      optimizer:tf.train.adam(this.learningRate),
      loss: tf.losses.meanSquaredError,
      metrics: ['accuracy']

    });
  
    this.modelCreated = true;
  }
 }
