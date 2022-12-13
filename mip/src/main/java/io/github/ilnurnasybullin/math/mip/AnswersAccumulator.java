package io.github.ilnurnasybullin.math.mip;

import io.github.ilnurnasybullin.math.simplex.SimplexAnswer;

interface AnswersAccumulator<T> {

    /**
     * Return true if accumulator has already better answer
     */
    boolean hasBetterThan(SimplexAnswer answer);
    boolean tryPutAnswer(SimplexAnswer answer);
    T answer();
}
