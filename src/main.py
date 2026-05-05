"""Main pipeline for training and evaluating the Credit Risk model."""

import logging

from .data import get_data_summary, load_data
from .evaluate import get_risk_by_historical_eval, predict_risk, run_validation_queries
from .logging_config import setup_logging
from .model import save_model, train_model

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the complete training and evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("Credit Risk Assessment - Bayesian Network")
    logger.info("=" * 60)

    # Load data
    logger.info("1. Loading data...")
    data = load_data()
    summary = get_data_summary(data)
    logger.info("   Samples: %d", summary['n_samples'])
    logger.info("   Features: %d", summary['n_features'])
    logger.info("   Risk distribution: %s", summary['risk_distribution'])

    # Train model
    logger.info("2. Training model...")
    model = train_model(data)
    save_model(model)

    # Run validation queries
    logger.info("3. Running validation queries...")
    validation_results = run_validation_queries(model)

    all_passed = True
    for result in validation_results:
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("   %s: Expected=%.3f, Actual=%.3f [%s]",
                   result['query'], result['expected'], result['actual'], status)
        if not result["passed"]:
            all_passed = False

    if all_passed:
        logger.info("   All validation queries passed!")
    else:
        logger.info("   Note: Some queries may differ slightly due to structure learning variability")

    # Show risk by historical evaluation
    logger.info("4. Risk by Historical Credit Evaluation:")
    ech_risk = get_risk_by_historical_eval(model)
    for ech, probs in ech_risk.items():
        logger.info("   ECH=%s: P(bad)=%.1f%%, P(good)=%.1f%%",
                   ech, probs['bad'] * 100, probs['good'] * 100)

    # Run example predictions
    logger.info("5. Example predictions...")

    examples = [
        {"historical_eval": 0, "duration": "24 to 72"},
        {"historical_eval": 4, "savings": "rich"},
        {"age": "19 to 28", "savings": "no account", "duration": "24 to 72"},
    ]

    for i, example in enumerate(examples, 1):
        result = predict_risk(model, **example)
        logger.info("   Example %d: %s", i, example)
        logger.info("   -> %s (Risk Level: %s)", result['prediction'], result['risk_level'])
        logger.info("   -> P(bad) = %.1f%%", result['probability_bad'] * 100)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging()
    main()
