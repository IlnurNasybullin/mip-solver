package io.github.ilnurnasybullin.math.mip;

import io.github.ilnurnasybullin.math.simplex.FunctionType;
import io.github.ilnurnasybullin.math.simplex.Simplex;
import io.github.ilnurnasybullin.math.simplex.SimplexAnswer;

import java.util.concurrent.locks.ReentrantLock;

class SingleAnswerAccumulator implements AnswersAccumulator<SimplexAnswer> {
    /**
     * Объект для блокировки к доступу (записи) некоторых ресурсов ({@link #recordValue}, {@link #answer})
     */
    private final ReentrantLock lock = new ReentrantLock();

    /**
     * Тип целевой функции. Необходим для выбора более оптимального значений функций из 2 предложенных (f<sub>1</sub> &lt
     * f<sub>2</sub>). При f &rarr min более оптимальное значение - f<sub>1</sub>, при f &rarr max - f<sub>2</sub>
     */
    private final FunctionType functionType;

    /**
     * Рекордное значение (наиболее оптимальное значение). До тех пор, пока не вычислено - его значение null.
     */
    private volatile Double recordValue;
    private volatile SimplexAnswer answer;

    SingleAnswerAccumulator(FunctionType functionType) {
        this.functionType = functionType;
    }

    @Override
    public boolean hasBetterThan(SimplexAnswer answer) {
        double fx = answer.fx();
        Double localRecordValue;
        lock.lock();
        localRecordValue = recordValue;
        lock.unlock();

        if (localRecordValue == null) {
            return false;
        }

        if (isApproximateEqual(fx, localRecordValue)) {
            return true;
        }

        return functionType == FunctionType.MAX ?
                fx <= localRecordValue  :
                fx >= localRecordValue;
    }

    private boolean isApproximateEqual(double x1, double x2) {
        return isApproximateEqual(x1, x2, Simplex.EPSILON);
    }

    private boolean isApproximateEqual(double x1, double x2, double epsilon) {
        return Math.abs(x1 - x2) < epsilon;
    }

    @Override
    public boolean tryPutAnswer(SimplexAnswer answer) {
        double fx = answer.fx();

        try {
            lock.lock();
            if (recordValue == null ||
                functionType == FunctionType.MAX && fx > recordValue ||
                functionType == FunctionType.MIN && fx < recordValue
            ) {
                updateResult(answer);
                return true;
            }
        } finally {
            lock.unlock();
        }

        return false;
    }

    private void updateResult(SimplexAnswer answer) {
        this.answer = answer;
        recordValue = answer.fx();
    }

    @Override
    public SimplexAnswer answer() {
        return answer;
    }
}
