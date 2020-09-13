/* 
 * File:   score_calculator.h
 * Author: malone
 *
 * Created on November 24, 2012, 5:46 PM
 */

#ifndef SCORE_CALCULATOR_H
#define	SCORE_CALCULATOR_H

#include "urlearning/base/typedefs.h"

#include "urlearning/score_cache/score_cache.h"

#include <boost/asio.hpp>


namespace scoring {
    
    class Constraints;
    class ScoringFunction;

    class ScoreCalculator {
    public:
        ScoreCalculator(scoring::ScoringFunction *scoringFunction, int maxParents, int variableCount, int runningTime, Constraints *constraints);

        ~ScoreCalculator() {
            // no pointers that are not deleted elsewhere
            // these could possibly be "smart pointers" or something...
        }
        
	// Ni added
	// neighbors is an in/out parameter. As an in parameter, it is the maximal set of parent/child variables  
	// As an out parameter, neighbors is the set of parent and child variables when the function returns
        void calculateScores(int variable, FloatMap &cache, varset & neighbors);
        void prune(FloatMap &cache);

    private:
	// Ni added
        // neighbors is an in/out parameter. As an in parameter, it is the maximal set of parent/child variables
        // As an out parameter, neighbors is the set of parent and child variables when the function returns
        void calculateScores_internal(int variable, FloatMap &cache, varset & neighbors);
        void timeout(const boost::system::error_code& /*e*/);
        
        int highestCompletedLayer;
        
        /**
         * A timer to keep track of how long the algorithm has been running.
         */
        boost::asio::io_service io;
        boost::asio::deadline_timer *t;

        /**
         * A variable to check if the user-specified time limit has expired.
         */
        bool outOfTime;
        
        ScoringFunction *scoringFunction;
        int maxParents;
        int variableCount;
        int runningTime;
        Constraints *constraints;
    };

}

#endif	/* SCORE_CALCULATOR_H */

