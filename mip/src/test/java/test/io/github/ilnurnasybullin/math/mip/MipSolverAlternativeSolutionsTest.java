package test.io.github.ilnurnasybullin.math.mip;

import io.github.ilnurnasybullin.math.mip.MipSolver;
import io.github.ilnurnasybullin.math.simplex.FunctionType;
import io.github.ilnurnasybullin.math.simplex.Inequality;
import io.github.ilnurnasybullin.math.simplex.Simplex;
import io.github.ilnurnasybullin.math.simplex.SimplexAnswer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MipSolverAlternativeSolutionsTest {
    @ParameterizedTest
    @MethodSource("_1_Data")
    public void test(Simplex.Builder builder, Set<double[]> expectedX, double expectedFx) {
        MipSolver simplex = new MipSolver();
        List<SimplexAnswer> answers = simplex.findAll(builder.build());
        answers.stream()
                .map(SimplexAnswer::X)
                .map(ArrayWrapper::new)
                .distinct()
                .peek(x -> System.out.printf("ANSWER IS: %s%n", x))
                .map(ArrayWrapper::array)
                .forEach(x -> Assertions.assertTrue(
                        removeArrayEquals(x, expectedX, Simplex.EPSILON),
                        errorMessage(x, expectedX)
                ));

        Assertions.assertTrue(expectedX.isEmpty());
    }

    private boolean removeArrayEquals(double[] checkingArray, Collection<double[]> arrays, double epsilon) {
        return arrays.removeIf(array -> arrayEquals(checkingArray, array, epsilon));
    }

    private boolean arrayEquals(double[] array1, double[] array2, double epsilon) {
        if (array1.length != array2.length) {
            return false;
        }

        for (int i = 0; i < array1.length; i++) {
            if (!isApproximateValue(array1[i], array2[i], epsilon)) {
                return false;
            }
        }

        return true;
    }

    private boolean isApproximateValue(double v1, double v2, double epsilon) {
        return Math.abs(v1 - v2) < epsilon;
    }

    private Supplier<String> errorMessage(double[] array, Collection<double[]> arrays) {
        return () -> String.format(
                "Expected that collection of arrays %s contains array %s",
                arrays.stream()
                        .map(Arrays::toString)
                        .collect(Collectors.joining(", ", "[", "]")),
                Arrays.toString(array)
        );
    }

    public static Stream<Arguments> _1_Data() {
        double[][] A = {
                {0.0, 1.0, 0.0, 0.0, 1.0, 1.0},
                {0.0, 0.0, 1.0, 1.0, 0.0, 1.0},
                {1.0, 0.0, 0.0, 1.0, 1.0, 0.0},
                {1.0, 1.0, 1.0, 0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
        };
        double[] B = {1, 1, 1, 1, 1};
        var inequalities = new Inequality[]{Inequality.LQ, Inequality.LQ, Inequality.LQ, Inequality.LQ, Inequality.LQ};

        // x + y -> MAX
        double[] C = {1, 1, 1, 1, 1, 1};
        var functionType = FunctionType.MAX;

        double[] X1 = {1, 0, 0, 0, 0, 0};
        double[] X2 = {0, 1, 0, 0, 0, 0};
        double[] X3 = {0, 0, 1, 0, 0, 0};
        double[] X4 = {0, 0, 0, 1, 0, 0};
        double[] X5 = {0, 0, 0, 0, 1, 0};
        double[] X6 = {0, 0, 0, 0, 0, 1};

        double fx = 1;

        Simplex.Builder builder = new Simplex.Builder()
                .setA(A)
                .setB(B)
                .setC(C)
                .setInequalities(inequalities)
                .setFunctionType(functionType);

        return Stream.of(Arguments.of(
                builder, new HashSet<>(Arrays.asList(X1, X2, X3, X4, X5, X6)), fx
        ));
    }

    private static class ArrayWrapper {

        private final double[] array;

        private ArrayWrapper(double[] array) {
            this.array = array;
        }

        public double[] array() {
            return Arrays.copyOf(array, array.length);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ArrayWrapper that = (ArrayWrapper) o;
            return Arrays.equals(array, that.array);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(array);
        }

        @Override
        public String toString() {
            return Arrays.toString(array);
        }
    }
}
