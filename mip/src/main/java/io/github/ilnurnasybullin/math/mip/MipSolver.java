package io.github.ilnurnasybullin.math.mip;

import io.github.ilnurnasybullin.math.simplex.Inequality;
import io.github.ilnurnasybullin.math.simplex.Simplex;
import io.github.ilnurnasybullin.math.simplex.SimplexAnswer;
import io.github.ilnurnasybullin.math.simplex.exception.SimplexDataException;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

import static io.github.ilnurnasybullin.math.simplex.Inequality.*;

/**
 * Класс, с помощью которого можно решать задачи дискретной оптимизации. Работает поверх класса {@link Simplex} с
 * использованием <a href=https://ru.wikipedia.org/wiki/Метод_ветвей_и_границ>алгоритма Лэнд и Дойг (метод ветвей и границ)</a>.
 * Каждый узел решается в отдельном потоке (для увеличения скорости вычисления)
 */
public class MipSolver {

    /**
     * Обработчик ошибок при решении задачи симплекс-методом (все возможные типы ошибок при решении задачи
     * симплекс-методом лежат в пакете {@link io.github.ilnurnasybullin.math.simplex.exception}. По умолчанию - вывод
     * в консоль {@link System#err} сообщения ошибки ({@link Throwable#getMessage()}). Для установки собственного
     * обработчика ошибок - воспользуйтесь методом {@link #exceptionHandler(Consumer)}
     */
    private Consumer<Throwable> exceptionHandler = exception -> System.err.println(exception.getMessage());

    private System.Logger logger = new NoOpsLogger();

    public MipSolver logger(System.Logger logger) {
        this.logger = logger;
        return this;
    }

    /**
     * Основной метод для взаимодействия пользователя с классом. В качестве аргумента передаётся настроенный
     * {@link Simplex.Builder} для решения задачи линейного программирования симплекс-методом, предикаты
     * ({@link Solver#correctValues}) и унарные операторы ({@link Solver#lowerBoundFunctions}, {@link Solver#upperBoundFunctions})
     * вычисляются по умолчанию, в качестве ответа возвращается {@link List<SimplexAnswer>}, содеражщий всевозможные
     * оптимальные значения вектора X ({@link SimplexAnswer#X()}) и значение функции ({@link SimplexAnswer#fx()}).
     * В процессе вычисления возможны все выбросы исключений при решении задачи симплекс-методом
     * ({@link io.github.ilnurnasybullin.math.simplex.exception}), которые, при решении во всех узлах (кроме корневого)
     * будут перехвачены обработчиком ошибок {@link #exceptionHandler}
     */

    public List<SimplexAnswer> findAll(Simplex simplex) {
        int xCount = simplex.xCount();

        return new FindAllSolver()
                .exceptionHandler(exceptionHandler)
                .upperBoundFunctions(defaultUpperBoundFunctions(xCount))
                .lowerBoundFunctions(defaultLowerBoundFunctions(xCount))
                .correctValues(defaultPredicates(xCount))
                .logger(logger)
                .solveSimplex(simplex);
    }

    public SimplexAnswer findAny(Simplex simplex) {
        int xCount = simplex.xCount();

        return new FindAnySolver()
                .exceptionHandler(exceptionHandler)
                .upperBoundFunctions(defaultUpperBoundFunctions(xCount))
                .lowerBoundFunctions(defaultLowerBoundFunctions(xCount))
                .correctValues(defaultPredicates(xCount))
                .logger(logger)
                .solveSimplex(simplex);
    }

    public static DoublePredicate[] defaultPredicates(int xCount) {
        DoublePredicate predicate = value -> {
            long longValue = (long) value;
            return isApproximateValue(value, longValue);
        };

        DoublePredicate[] predicates = new DoublePredicate[xCount];
        Arrays.fill(predicates, predicate);

        return predicates;
    }

    private static boolean isApproximateValue(double value, double actual) {
        return Math.abs(value - actual) < Simplex.EPSILON;
    }

    public static DoubleUnaryOperator[] defaultUpperBoundFunctions(int xCount) {
        DoubleUnaryOperator operator = Math::ceil;

        DoubleUnaryOperator[] operators = new DoubleUnaryOperator[xCount];
        Arrays.fill(operators, operator);

        return operators;
    }

    public static DoubleUnaryOperator[] defaultLowerBoundFunctions(int xCount) {
        DoubleUnaryOperator operator = Math::floor;

        DoubleUnaryOperator[] operators = new DoubleUnaryOperator[xCount];
        Arrays.fill(operators, operator);

        return operators;
    }

    private <T> void validateArrayLength(int length, T[] array) {
        if (array == null || array.length != length) {
            throw new SimplexDataException(String.format("Invalid data: (%s) for this array!", Arrays.toString(array)));
        }
    }

    public MipSolver exceptionHandler(Consumer<Throwable> exceptionHandler) {
        this.exceptionHandler = exceptionHandler;
        return this;
    }

    interface Solver<T> {
        Solver<T> logger(System.Logger logger);
        Solver<T> correctValues(DoublePredicate[] correctValues);
        Solver<T> lowerBoundFunctions(DoubleUnaryOperator[] lowerBoundFunctions);
        Solver<T> upperBoundFunctions(DoubleUnaryOperator[] upperBoundFunctions);
        Solver<T> exceptionHandler(Consumer<Throwable> exceptionHandler);

        T solveSimplex(Simplex simplex);
    }

    private static abstract class SimpleSolver<T> implements Solver<T> {

        /**
         * Предикаты (в количестве {@link Simplex#xCount()}), которые определяют, является ли x<sub>i</sub> корректным значением,
         * удовлетворяющим условию задачи дискретной оптимизации (i = 1 .. {@link Simplex#xCount()}). По умолчанию, используются
         * предикаты {@link #defaultPredicates(int)}, которые сравнивают число double (вещественное число) с приведённым long
         * (целым числом) c точностью до {@link Simplex#EPSILON}
         */
        protected DoublePredicate[] correctValues;

        /**
         * Функции (унарные операторы) в количестве {@link Simplex#xCount()}, которые преобразуют x<sub>i</sub> в некоторое значение
         * d<sub>i</sub> : d<sub>i</sub> &le x<sub>i</sub>, d<sub>i</sub> &isin D, i = 1 .. {@link Simplex#xCount()}, D - дискретное
         * множество, удовлетворяющее условию задачи дискретной оптимизации. По умолчанию, используются унарные операторы
         * {@link #defaultLowerBoundFunctions(int)}, которые возвращают ближайшее "целое" число (представленное типом double,
         * {@link Math#floor(double)}), меньшее числа x<sub>i</sub>
         */
        protected DoubleUnaryOperator[] lowerBoundFunctions;

        /**
         * Функции (унарные операторы) в количестве {@link Simplex#xCount()}, которые преобразуют x<sub>i</sub> в некоторое значение
         * e<sub>i</sub> : e<sub>i</sub> &ge x<sub>i</sub>, e<sub>i</sub> &isin D, i = 1 .. {@link Simplex#xCount()}, D - дискретное
         * множество, удовлетворяющее условию задачи дискретной оптимизации. По умолчанию, используются унарные операторы
         * {@link #defaultUpperBoundFunctions(int)}, которые возвращают ближайшее "целое" число (представленное типом double,
         * {@link Math#ceil(double)}), большее числа x<sub>i</sub>
         */
        protected DoubleUnaryOperator[] upperBoundFunctions;

        protected Consumer<Throwable> exceptionHandler = exception -> System.err.println(exception.getMessage());

        protected System.Logger logger = new NoOpsLogger();

        @Override
        public SimpleSolver<T> logger(System.Logger logger) {
            this.logger = logger;
            return this;
        }

        @Override
        public SimpleSolver<T> correctValues(DoublePredicate[] correctValues) {
            this.correctValues = correctValues;
            return this;
        }

        @Override
        public SimpleSolver<T> lowerBoundFunctions(DoubleUnaryOperator[] lowerBoundFunctions) {
            this.lowerBoundFunctions = lowerBoundFunctions;
            return this;
        }

        @Override
        public SimpleSolver<T> upperBoundFunctions(DoubleUnaryOperator[] upperBoundFunctions) {
            this.upperBoundFunctions = upperBoundFunctions;
            return this;
        }

        @Override
        public SimpleSolver<T> exceptionHandler(Consumer<Throwable> exceptionHandler) {
            this.exceptionHandler = exceptionHandler;
            return this;
        }

        protected abstract AnswersAccumulator<T> answersAccumulator(Simplex simplex);

        @Override
        public T solveSimplex(Simplex simplex) {
            AnswersAccumulator<T> answersAccumulator = answersAccumulator(simplex);
            Queue<SimplexWrapper> simplexes = new ArrayDeque<>();
            simplexes.add(new SimpleWrapper(simplex));

            while (!simplexes.isEmpty()) {
                SimplexWrapper wrapper = simplexes.remove();

                SimplexAnswer answer = null;
                try {
                    Simplex smp = wrapper.solve();
                    answer = smp.solve();
                } catch (Exception e) {
                    exceptionHandler.accept(e);
                    continue;
                }

                if (isLegalAnswer(answer)) {
                    answersAccumulator.tryPutAnswer(answer);
                    continue;
                }

                if (!answersAccumulator.hasBetterThan(answer)) {
                    List<SimplexWrapper> newSimplexes = createWithNewConstraints(wrapper);
                    simplexes.addAll(newSimplexes);
                }
            }

            return answersAccumulator.answer();
        }

        protected List<SimplexWrapper> createWithNewConstraints(SimplexWrapper wrapper) {
            double[] X =  wrapper.simplex().solve().X();
            int xIndex = getInvalidXIndex(X);

            double lowerBound = lowerBoundFunctions[xIndex].applyAsDouble(X[xIndex]);
            SimplexWrapper lower = wrapper.lowerBoundX(xIndex, lowerBound);

            double upperBound = upperBoundFunctions[xIndex].applyAsDouble(X[xIndex]);
            SimplexWrapper upper = wrapper.upperBoundX(xIndex, upperBound);

            return List.of(lower, upper);
        }

        protected boolean isLegalAnswer(SimplexAnswer answer) {
            return getInvalidXIndex(answer.X()) == null;
        }

        protected Integer getInvalidXIndex(double[] x) {
            for (int i = 0; i < x.length; i++) {
                if (!correctValues[i].test(x[i])) {
                    return i;
                }
            }

            return null;
        }

    }

    private static class FindAnySolver extends SimpleSolver<SimplexAnswer> {

        @Override
        protected AnswersAccumulator<SimplexAnswer> answersAccumulator(Simplex simplex) {
            return new SingleAnswerAccumulator(simplex.functionType());
        }
    }

    private static class FindAllSolver extends SimpleSolver<List<SimplexAnswer>> {

        @Override
        protected AnswersAccumulator<List<SimplexAnswer>> answersAccumulator(Simplex simplex) {
            return new MultiAnswersAccumulator(simplex.functionType());
        }

        @Override
        public List<SimplexAnswer> solveSimplex(Simplex simplex) {
            AnswersAccumulator<List<SimplexAnswer>> answersAccumulator = answersAccumulator(simplex);
            Queue<SimplexWrapper> simplexes = new ArrayDeque<>();
            simplexes.add(new SimpleWrapper(simplex));

            while (!simplexes.isEmpty()) {
                SimplexWrapper wrapper = simplexes.remove();

                SimplexAnswer answer = null;
                try {
                    Simplex smp = wrapper.solve();
                    answer = smp.solve();

                    if (!wrapper.isAlternativeSolution()) {
                        logger.log(System.Logger.Level.INFO, "FINDING ALTERNATIVE SOLUTIONS...");
                        int size = simplexes.size();
                        smp.findAlternativeSolutions()
                                .stream()
                                .map(wrapper::alternativeSolution)
                                .forEach(simplexes::add);
                        logger.log(System.Logger.Level.INFO, "ALTERNATIVE SOLUTIONS ARE FOUNDED, size is %d%n", simplexes.size() - size);
                    }
                } catch (Exception e) {
                    exceptionHandler.accept(e);
                    continue;
                }

                if (isLegalAnswer(answer)) {
                    answersAccumulator.tryPutAnswer(answer);
                    continue;
                }

                if (!answersAccumulator.hasBetterThan(answer)) {
                    List<SimplexWrapper> newSimplexes = createWithNewConstraints(wrapper);
                    simplexes.addAll(newSimplexes);
                }
            }

            return answersAccumulator.answer();
        }
    }

    interface SimplexWrapper {
        SimplexWrapper lowerBoundX(int index, double bi);
        SimplexWrapper upperBoundX(int index, double bi);
        Simplex solve();
        Simplex simplex();

        SimplexWrapper alternativeSolution(Simplex simplex);
        boolean isAlternativeSolution();

        enum BoundType {
            LOWER,
            UPPER
        }
    }

    private static class AlternativeWrapper implements SimplexWrapper {

        private final SimplexWrapper wrapper;

        private AlternativeWrapper(SimplexWrapper wrapper) {
            this.wrapper = wrapper;
        }

        @Override
        public SimplexWrapper lowerBoundX(int index, double bi) {
            return wrapper.lowerBoundX(index, bi);
        }

        @Override
        public SimplexWrapper upperBoundX(int index, double bi) {
            return wrapper.upperBoundX(index, bi);
        }

        @Override
        public Simplex solve() {
            return wrapper.solve();
        }

        @Override
        public Simplex simplex() {
            return wrapper.simplex();
        }

        @Override
        public SimplexWrapper alternativeSolution(Simplex simplex) {
            return new AlternativeWrapper(wrapper.alternativeSolution(simplex));
        }

        @Override
        public boolean isAlternativeSolution() {
            return true;
        }
    }

    private static class SimpleWrapper implements SimplexWrapper {

        private final Simplex simplex;

        private SimpleWrapper(Simplex simplex) {
            this.simplex = simplex.copy();
        }

        @Override
        public SimplexWrapper lowerBoundX(int index, double bi) {
            return new AddConstraintWrapper(simplex, index, bi, BoundType.LOWER);
        }

        @Override
        public SimplexWrapper upperBoundX(int index, double bi) {
            return new AddConstraintWrapper(simplex, index, bi, BoundType.UPPER);
        }

        @Override
        public Simplex solve() {
            simplex.solve();
            return simplex;
        }

        @Override
        public Simplex simplex() {
            return simplex;
        }

        @Override
        public SimplexWrapper alternativeSolution(Simplex simplex) {
            return new AlternativeWrapper(new SimpleWrapper(simplex));
        }

        @Override
        public boolean isAlternativeSolution() {
            return false;
        }
    }

    private static class AddConstraintWrapper implements SimplexWrapper {

        private final Simplex simplex;
        private final int xIndex;
        private final double bi;

        private final BoundType boundType;
        private final Map<Integer, Integer> lowerBoundIndexes;
        private final Map<Integer, Integer> upperBoundIndexes;

        private AddConstraintWrapper(Simplex simplex, int xIndex, double bi, BoundType boundType) {
            this(simplex, xIndex, bi, boundType, Map.of(), Map.of());
        }

        private AddConstraintWrapper(Simplex simplex, int xIndex, double bi, BoundType boundType,
                                     Map<Integer, Integer> lowerBoundIndexes, Map<Integer, Integer> upperBoundIndexes) {
            this.simplex = simplex.copy();
            this.xIndex = xIndex;
            this.bi = bi;
            this.boundType = boundType;
            this.lowerBoundIndexes = new HashMap<>(lowerBoundIndexes);
            this.upperBoundIndexes = new HashMap<>(upperBoundIndexes);

            if (boundType == BoundType.LOWER) {
                this.lowerBoundIndexes.put(xIndex, simplex.bConstraintsCount());
            } else {
                this.upperBoundIndexes.put(xIndex, simplex.bConstraintsCount());
            }
        }

        @Override
        public SimplexWrapper lowerBoundX(int index, double bi) {
            if (lowerBoundIndexes.containsKey(index)) {
                return new ChangeBWrapper(simplex, index, bi, BoundType.LOWER, lowerBoundIndexes, upperBoundIndexes);
            }

            return new AddConstraintWrapper(simplex, index, bi, BoundType.LOWER, lowerBoundIndexes, upperBoundIndexes);
        }

        @Override
        public SimplexWrapper upperBoundX(int index, double bi) {
            if (upperBoundIndexes.containsKey(index)) {
                return new ChangeBWrapper(simplex, index, bi, BoundType.UPPER, lowerBoundIndexes, upperBoundIndexes);
            }

            return new AddConstraintWrapper(simplex, index, bi, BoundType.UPPER, lowerBoundIndexes, upperBoundIndexes);
        }

        @Override
        public Simplex solve() {
            double[] ai = new double[simplex.xCount()];
            ai[xIndex] = 1;

            Inequality inequality =
                    boundType == BoundType.LOWER ?
                            LQ : GE;
            simplex.addConstraint(ai, inequality, bi);
            return simplex;
        }

        @Override
        public Simplex simplex() {
            return simplex;
        }

        @Override
        public SimplexWrapper alternativeSolution(Simplex simplex) {
            return new AlternativeWrapper(new AddConstraintWrapper(simplex, xIndex, bi, boundType, lowerBoundIndexes, upperBoundIndexes));
        }

        @Override
        public boolean isAlternativeSolution() {
            return false;
        }
    }

    private static class ChangeBWrapper implements SimplexWrapper {

        private final Simplex simplex;
        private final int bIndex;
        private final double bi;

        private final Map<Integer, Integer> lowerBoundIndexes;
        private final Map<Integer, Integer> upperBoundIndexes;

        private ChangeBWrapper(Simplex simplex, int xIndex, double bi, BoundType boundType,
                               Map<Integer, Integer> lowerBoundIndexes, Map<Integer, Integer> upperBoundIndexes) {
            this.simplex = simplex.copy();
            this.bIndex = boundType == BoundType.LOWER ? lowerBoundIndexes.get(xIndex) : upperBoundIndexes.get(xIndex);
            this.bi = bi;
            this.lowerBoundIndexes = Map.copyOf(lowerBoundIndexes);
            this.upperBoundIndexes = Map.copyOf(upperBoundIndexes);
        }

        private ChangeBWrapper(Simplex simplex, int bIndex, double bi,
                               Map<Integer, Integer> lowerBoundIndexes, Map<Integer, Integer> upperBoundIndexes) {
            this.simplex = simplex.copy();
            this.bIndex = bIndex;
            this.bi = bi;
            this.lowerBoundIndexes = Map.copyOf(lowerBoundIndexes);
            this.upperBoundIndexes = Map.copyOf(upperBoundIndexes);
        }

        @Override
        public SimplexWrapper lowerBoundX(int index, double bi) {
            if (lowerBoundIndexes.containsKey(index)) {
                return new ChangeBWrapper(simplex, index, bi, BoundType.LOWER, lowerBoundIndexes, upperBoundIndexes);
            }

            return new AddConstraintWrapper(simplex, index, bi, BoundType.LOWER, lowerBoundIndexes, upperBoundIndexes);
        }

        @Override
        public SimplexWrapper upperBoundX(int index, double bi) {
            if (upperBoundIndexes.containsKey(index)) {
                return new ChangeBWrapper(simplex, index, bi, BoundType.UPPER, lowerBoundIndexes, upperBoundIndexes);
            }

            return new AddConstraintWrapper(simplex, index, bi, BoundType.UPPER, lowerBoundIndexes, upperBoundIndexes);
        }

        @Override
        public Simplex solve() {
            simplex.changeB(bIndex, bi);
            return simplex;
        }

        @Override
        public Simplex simplex() {
            return simplex;
        }

        @Override
        public SimplexWrapper alternativeSolution(Simplex simplex) {
            return new AlternativeWrapper(new ChangeBWrapper(simplex, bIndex, bi, lowerBoundIndexes, upperBoundIndexes));
        }

        @Override
        public boolean isAlternativeSolution() {
            return false;
        }
    }

}
