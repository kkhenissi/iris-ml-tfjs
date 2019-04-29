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

  ngOnInit(): void {
    this.tensors[ 'X'] = tf.tensor2d([45, 11, 25, 31, 45, 26, 45, 21, 22, 15, 44, 66], [3, 4]);
    this.tensors[ 'Y'] = tf.tensor2d([[5, 44, 7], [5, 15, 21], [74, 58, 47], [95, 42, 84] ]);
    this.tensors['X'].print();
    this.tensors['Y'].print();

  }

  onMult() {
    this.tensors['Z'] = this.tensors['X'].matMul(this.tensors['Y']);
  }
}
